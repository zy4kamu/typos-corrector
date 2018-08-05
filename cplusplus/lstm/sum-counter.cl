__kernel void calculate_sum(__global float* data, int size)
{
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    uint local_size = get_local_size(0);

    for (int shift = local_size / 2; shift > 0; shift >>= 1) {
        if (local_id < shift) {
            data[global_id] += data[global_id + shift];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_id == 0) {
        for (uint i = local_size; i < size; i += local_size) {
            data[0] += data[i];
        }
    }
}

__kernel void calculate_exp(__global float* data, int size)
{
    size_t global_id = get_global_id(0);
    data[global_id] = exp(data[global_id]);
}
