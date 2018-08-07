#include "compressed_lstm.h"

#include <cassert>
#include <fstream>
#include <clBLAS.h>

namespace {

std::vector<cl_float> read_file(const std::string& filename) {
    std::vector<cl_float> data;
    std::ifstream file(filename, std::ios::binary);
    cl_float item;
    while (file.read(reinterpret_cast<char*>(&item), sizeof(cl_float))) {
        data.push_back(item);
    }
    return data;
}

void read_buffer_from_file(cl::CommandQueue& queue, const std::string& input_file,
                           cl::Buffer& buffer, size_t size) {
    std::vector<cl_float> data = read_file(input_file);
    assert(data.size() == size);
    int error = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(cl_float) * size, data.data());
    assert(error == 0);
}

} // anonymous namespace

CompressedLSTMCell::CompressedLSTMCell(const std::string& input_folder, cl_int input_size, cl_int compressor_size, cl_int lstm_size)
    : input_size(input_size), compressor_size(compressor_size), lstm_size(lstm_size) {
    // get platform
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);

    // get device
    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    device = devices.front();

    // get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/compressed_lstm.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // build program
    context = cl::Context(device);
    program = cl::Program(context, sources);
    int error = program.build();
    assert(error == 0);
    queue = cl::CommandQueue(context, device);

    // create buffers
    input_and_hidden_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                         sizeof(cl_float) * (input_size + lstm_size));
    state_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * lstm_size);
    left_matrix_kernel_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                           sizeof(cl_float) * (input_size + lstm_size) * compressor_size);
    intermediate_matrix_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                           sizeof(cl_float) * compressor_size);
    right_matrix_kernel_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                            sizeof(cl_float) * compressor_size * 4 * lstm_size);
    bias_kernel_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_float) * lstm_size);
    ijfo_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_float) * lstm_size);

    // make all buffers zero
    reset();

    // create kernel for lstm cell computation
    lstm_cell_kernel = cl::Kernel(program, "lstm_cell", &error);
    assert(error == 0);
    lstm_cell_kernel.setArg(0, input_and_hidden_buffer);
    lstm_cell_kernel.setArg(1, state_buffer);
    lstm_cell_kernel.setArg(2, ijfo_buffer);
    lstm_cell_kernel.setArg(3, input_size);
    lstm_cell_kernel.setArg(4, lstm_size);

    // read parameters from file
    read_buffer_from_file(queue, input_folder + "/bias", bias_kernel_buffer,
                          4 * lstm_size);
    read_buffer_from_file(queue, input_folder + "/left_matrix", left_matrix_kernel_buffer,
                          (input_size + lstm_size) * compressor_size);
    read_buffer_from_file(queue, input_folder + "/right_matrix", right_matrix_kernel_buffer,
                          compressor_size * 4 * lstm_size);

    // setup clBLAS
    error = clblasSetup();
    assert(error == CL_SUCCESS);
}

void CompressedLSTMCell::reset() {
    int error = 0;
    cl::Kernel initialize_buffers_kernel(program, "initialize", &error);
    assert(error == 0);
    initialize_buffers_kernel.setArg(0, input_and_hidden_buffer);
    initialize_buffers_kernel.setArg(1, state_buffer);
    initialize_buffers_kernel.setArg(2, input_size);
    initialize_buffers_kernel.setArg(3, lstm_size);
    error = queue.enqueueNDRangeKernel(initialize_buffers_kernel, 0, lstm_size, 1);
    assert(error == 0);
}

void CompressedLSTMCell::process(const std::vector<cl_float>& input, std::vector<cl_float>& output) {
    assert(static_cast<cl_float>(input.size()) == input_size);
    assert(static_cast<cl_float>(output.size()) == lstm_size);

    // calculate ijfo matrix (the only place where matrix multiplication is done)
    calculate_ijfo(input);

    // calculate hidden_buffer and state_buffer
    cl::Event event;
    int error = queue.enqueueNDRangeKernel(lstm_cell_kernel, 0, lstm_size, 1, NULL, &event);
    assert(error == 0);
    event.wait();

    // read hidden buffer back
    error = queue.enqueueReadBuffer(input_and_hidden_buffer, CL_TRUE, sizeof(cl_float) * input_size, sizeof(cl_float) * lstm_size,
                                    output.data());
    assert(error == 0);
}

void CompressedLSTMCell::calculate_ijfo(const std::vector<cl_float>& input) {
    // copy input to input_and_hidden_buffer
    int error = queue.enqueueWriteBuffer(input_and_hidden_buffer, CL_FALSE, 0, sizeof(cl_float) * input_size, input.data());
    assert(error == 0);

    // multiply on left matrix
    cl_command_queue local_queue =  queue.get();
    clblasStatus status =
    clblasSgemv(clblasRowMajor,                   // order
                clblasTrans,                      // transA
                input_size + lstm_size,           // M
                compressor_size,                  // N
                1.f,                              // alpha
                left_matrix_kernel_buffer.get(),  // A
                0,                                // offA
                compressor_size,                  // lda
                input_and_hidden_buffer.get(),    // x
                0,                                // offx
                1,                                // incx
                0,                                // beta
                intermediate_matrix_buffer.get(), // y
                0,                                // offy
                1,                                // incy
                1,                                // numCommandQueues
                &local_queue,                     // commandQueues
                0,                                // numEventsInWaitList
                NULL,                             // eventWaitList
                NULL);                            // events
    assert(status == clblasSuccess);

    // multiply on right matrix
    status =
    clblasSgemv(clblasRowMajor,                   // order
                clblasTrans,                      // transA
                compressor_size,                  // M
                4 * lstm_size,                    // N
                1.f,                              // alpha
                right_matrix_kernel_buffer.get(), // A
                0,                                // offA
                4 * lstm_size,                    // lda
                intermediate_matrix_buffer.get(), // x
                0,                                // offx
                1,                                // incx
                0,                                // beta
                ijfo_buffer.get(),                // y
                0,                                // offy
                1,                                // incy
                1,                                // numCommandQueues
                &local_queue,                     // commandQueues
                0,                                // numEventsInWaitList
                NULL,                             // eventWaitList
                NULL);                            // events
    assert(status == clblasSuccess);

    // add bias
    status =
    clblasSaxpy(4 * lstm_size,            // N
                1,                        // alpha
                bias_kernel_buffer.get(), // X
                0,                        // offx
                1,                        // incx
                ijfo_buffer.get(),        // Y
                0,                        // offy
                1,                        // incy
                1,                        // numCommandQueues
                &local_queue,             // commandQueues
                0,                        // numEventsInWaitList
                NULL,                     // eventWaitList
                NULL);                    // events
}
