#pragma once

#include <cl2.hpp>
#include <vector>

#include <boost/filesystem/path.hpp>

struct OpenCLConnector {
    OpenCLConnector();
    cl::Buffer read_buffer_from_file(const boost::filesystem::path& input_file, size_t size, int memory_permissions);
    void vector_matrix_multiply(const cl::Buffer& vector, const cl::Buffer& matrix, cl_int matrix_num_rows,
                                cl_int matrix_num_cols, cl::Buffer& output);
    void add_to_vector(const cl::Buffer& to_add, cl::Buffer& vector, cl_int size);

    std::vector<cl::Platform> platforms;
    cl::Device                device;
    cl::Context               context;
    cl::CommandQueue          queue;
};
