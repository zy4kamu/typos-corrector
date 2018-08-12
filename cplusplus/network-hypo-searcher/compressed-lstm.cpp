#include "compressed-lstm.h"

#include <cassert>
#include <fstream>

#include "common.h"
#include "../utils/utils.h"

namespace NOpenCLConnector {

CompressedLSTMCell::CompressedLSTMCell(OpenCLConnector& opencl_connector,
                                       const boost::filesystem::path& input_folder,
                                       const std::vector<std::string>& models)
    : opencl_connector(opencl_connector) {

    // read parameters from file
    for (size_t i = 0; i < models.size(); ++i) {
        //  get sizes
        const std::string model_prefix = (input_folder / models[i]).string();
        int_type local_lstm_size = static_cast<int_type>(get_file_size(model_prefix + "bias")) / (4 * sizeof(float_type));
        int_type local_compressor_size = static_cast<int_type>(get_file_size(model_prefix + "right_matrix") /
                                                           (4 * sizeof(float_type) * local_lstm_size));
        int_type local_input_size = static_cast<int_type>(get_file_size(model_prefix + "left_matrix") /
                                                      (sizeof(float_type) * local_compressor_size)) - local_lstm_size;
        if (i == 0) {
            lstm_size = local_lstm_size;
            compressor_size = local_compressor_size;
            input_size = local_input_size;

            // create buffers
            // TODO: experiment with memory access types
            input_and_hidden_buffer = cl::Buffer(opencl_connector.context, CL_MEM_READ_WRITE,
                                                 sizeof(float_type) * (input_size + lstm_size));
            state_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, sizeof(float_type) * lstm_size);
            ijfo_buffer = cl::Buffer(opencl_connector.context, CL_MEM_WRITE_ONLY, 4 * sizeof(float_type) * lstm_size);

            // hidden_buffer is a sub buffer of input_and_hidden_buffer and is created for simplicity
            hidden_buffer_region.origin = sizeof(float_type) * input_size;
            hidden_buffer_region.size = sizeof(float_type) * lstm_size;
            hidden_buffer = input_and_hidden_buffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                                                    &hidden_buffer_region);

            // hidden_buffer is a sub buffer of input_and_hidden_buffer and is created for simplicity
            input_buffer_region.origin = 0;
            input_buffer_region.size = sizeof(float_type) * input_size;
            input_buffer = input_and_hidden_buffer.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                                                   &input_buffer_region);
        } else {
            assert(lstm_size == local_lstm_size);
            assert(compressor_size == local_compressor_size);
            assert(input_size == local_input_size);
        }

        left_matrix_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "left_matrix",
                                                                                (input_size + lstm_size) * compressor_size,
                                                                                CL_MEM_WRITE_ONLY));
        intermediate_matrix_buffers.emplace_back(opencl_connector.context, CL_MEM_WRITE_ONLY,
                                                 sizeof(float_type) * compressor_size);
        right_matrix_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "right_matrix",
                                                                                compressor_size * 4 * lstm_size,
                                                                                CL_MEM_WRITE_ONLY));
        bias_buffers.emplace_back(opencl_connector.read_buffer_from_file(model_prefix + "bias",
                                                                                4 * lstm_size,
                                                                                CL_MEM_WRITE_ONLY));
    }

    // get source code
    std::ifstream reader(std::string(ROOT_DIRECTORY) + "/compressed-lstm.cl");
    std::string src(std::istreambuf_iterator<char>(reader), (std::istreambuf_iterator<char>()));
    assert(src.size() > 0);
    sources = cl::Program::Sources(1, src);

    // build program
    program = cl::Program(opencl_connector.context, sources);
    int error = program.build();
    assert(error == 0);

    // create kernel for lstm cell computation
    lstm_cell_kernel = cl::Kernel(program, "lstm_cell", &error);
    assert(error == 0);
    lstm_cell_kernel.setArg(0, hidden_buffer);
    lstm_cell_kernel.setArg(1, state_buffer);
    lstm_cell_kernel.setArg(2, ijfo_buffer);
    lstm_cell_kernel.setArg(3, input_size);
    lstm_cell_kernel.setArg(4, lstm_size);

    // create buffer for reseting lstm after each start
    error = 0;
    initialize_buffers_kernel = cl::Kernel(program, "reset", &error);
    assert(error == 0);
    initialize_buffers_kernel.setArg(0, hidden_buffer);
    initialize_buffers_kernel.setArg(1, state_buffer);
    initialize_buffers_kernel.setArg(2, input_size);
    initialize_buffers_kernel.setArg(3, lstm_size);

    // create buffer for setup input on each lstm_cell call
    error = 0;
    set_input_kernel = cl::Kernel(program, "set_input", &error);
    assert(error == 0);
    set_input_kernel.setArg(0, input_buffer);

    // make all buffers zero
    reset();
}

void CompressedLSTMCell::reset() {
    int error = opencl_connector.queue.enqueueNDRangeKernel(initialize_buffers_kernel, 0, lstm_size, 1);
    assert(error == 0);
}

void CompressedLSTMCell::get_output(std::vector<float_type>& output) const {
    assert(static_cast<float_type>(output.size()) == lstm_size);
    int error = opencl_connector.queue.enqueueReadBuffer(hidden_buffer, CL_TRUE, 0, sizeof(float_type) * lstm_size,
                                                         output.data());
    assert(error == 0);
}

void CompressedLSTMCell::process(size_t one_hot_index, size_t model_index) {
    calculate_ijfo(one_hot_index, model_index);
    int error = opencl_connector.queue.enqueueNDRangeKernel(lstm_cell_kernel, 0, lstm_size, 1);
    assert(error == 0);
}

void CompressedLSTMCell::calculate_ijfo(int_type one_hot_index, size_t model_index) {
    assert(model_index < left_matrix_buffers.size());
    cl::Buffer& left_matrix_buffer = left_matrix_buffers[model_index];
    cl::Buffer& intermediate_matrix_buffer = intermediate_matrix_buffers[model_index];
    cl::Buffer& right_matrix_buffer = right_matrix_buffers[model_index];
    cl::Buffer& bias_buffer = bias_buffers[model_index];

    // set input
    set_input_kernel.setArg(1, one_hot_index);
    int error = opencl_connector.queue.enqueueNDRangeKernel(set_input_kernel, 0, input_size, 1);
    assert(error == 0);

    // multiply on left matrix
    opencl_connector.vector_matrix_multiply(input_and_hidden_buffer, left_matrix_buffer, input_size + lstm_size,
                                            compressor_size, intermediate_matrix_buffer);

    // multiply on right matrix
    opencl_connector.vector_matrix_multiply(intermediate_matrix_buffer, right_matrix_buffer,
                                            compressor_size, 4 * lstm_size, ijfo_buffer);

    // add bias
    opencl_connector.add_to_vector(bias_buffer, ijfo_buffer, 4 * lstm_size);
}

} // namespace NOpenCLConnector
