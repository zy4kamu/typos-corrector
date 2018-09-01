__kernel void intermediate_multipilcation(__global float* vector, __global float* matrix, int num_rows, int num_cols,
                                          __global float* output) {
    int global_id = get_global_id(0);
    int row_index = global_id / num_cols;
    int col_index = global_id % num_cols;
    vector += 32 * row_index;
    matrix += 32 * row_index * num_cols + col_index;
    float value = 0;
    for (int i = 0; i < 32; ++i) {
        value += vector[i] * matrix[i * num_cols];
    }
    output[row_index * num_cols + col_index] = value;
}

__kernel void final_sum(__global float* buffer, int num_rows, int num_cols, __global float* output) {
    int col_index = get_global_id(0);
    float value = 0;
    for (int i = 0; i < num_rows; ++i) {
        value += buffer[i * num_cols + col_index];
    }
    output[col_index] = value;
}
