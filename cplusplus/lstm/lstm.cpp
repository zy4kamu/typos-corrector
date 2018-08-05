#include "lstm.h"

#include <cassert>
#include <fstream>

LSTMCell::LSTMCell(cl_int input_size, cl_int lstm_size)
    : input_size(input_size), lstm_size(lstm_size) {
    // Get platform
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);

    // Get device
    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(devices.size() > 0);
    device = devices.front();

    // Get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/lstm.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // Build program
    context = cl::Context(device);
    program = cl::Program(context, sources);
    int error = program.build();
    assert(error == 0);

    // Create buffers
    input_and_hidden_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                         sizeof(cl_float) * (input_size + lstm_size));
    state_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * lstm_size);
    matrix_kernel_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                      4 * sizeof(cl_float) * lstm_size * (input_size + lstm_size));
    bias_kernel_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                                    4 * sizeof(cl_float) * (input_size + lstm_size));
    ijfo_buffer = cl::Buffer(context, CL_MEM_READ_WRITE,
                      4 * sizeof(cl_float) * (input_size + lstm_size));
    lstm_cell_kernel = cl::Kernel(program, "lstm_cell", &error);
    assert(error == 0);
    lstm_cell_kernel.setArg(0, input_and_hidden_buffer);
    lstm_cell_kernel.setArg(1, state_buffer);
    lstm_cell_kernel.setArg(2, ijfo_buffer);
    lstm_cell_kernel.setArg(3, input_size);
    lstm_cell_kernel.setArg(4, lstm_size);

    // TODO: make all buffers zero

    // Create exp_kernel
    device_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * lstm_size);
    exp_kernel = cl::Kernel(program, "calculate_exp", &error);
    assert(error == 0);
    exp_kernel.setArg(0, device_buffer);
    exp_kernel.setArg(1, lstm_size);

    queue = cl::CommandQueue(context, device);
}

std::vector<cl_float>& LSTMCell::exp(std::vector<cl_float>& data) {
    assert(static_cast<cl_float>(data.size()) == lstm_size);
    int error = queue.enqueueWriteBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float) * lstm_size, data.data());
    assert(error == 0);
    error = queue.enqueueNDRangeKernel(exp_kernel, 0, lstm_size, 1, NULL);
    assert(error == 0);
    error = queue.enqueueReadBuffer(device_buffer, CL_TRUE, 0, sizeof(cl_float) * lstm_size, data.data());
    assert(error == 0);
    return data;
}

void LSTMCell::process(const std::vector<cl_float>& input, std::vector<cl_float>& output) {
    assert(static_cast<cl_float>(input.size()) == input_size);
    assert(static_cast<cl_float>(output.size()) == lstm_size);

    // copy input to input_and_hidden_buffer
    int error = queue.enqueueWriteBuffer(input_and_hidden_buffer, CL_TRUE, 0, sizeof(cl_float) * input_size, input.data());
    assert(error == 0);

    // TODO: calculate ijfo with clBLAS

    // calculate hidden_buffer and state_buffer
    error = queue.enqueueNDRangeKernel(lstm_cell_kernel, 0, lstm_size, 1, NULL);
    assert(error == 0);

    // read hidden buffer back
    error = queue.enqueueReadBuffer(input_and_hidden_buffer, CL_TRUE, input_size, sizeof(cl_float) * lstm_size,
                                    output.data());
    assert(error == 0);
}
