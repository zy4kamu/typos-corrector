__kernel void initialize_logits(__global float* logits) {
    size_t global_id = get_global_id(0);
    logits[global_id] = -1024;
}

// NOTE: this kernel must be called with 1 local group of size 32
__kernel void logits_to_probabilities(__global float* buffer) {
    size_t global_id = get_global_id(0);

    // create local buffer
    __local float sum_buffer[32];

    // exponent all logits
    buffer[global_id] = exp(buffer[global_id]);

    // get normalization factor
    sum_buffer[global_id] = buffer[global_id];
    for (int shift = 16; shift > 0; shift >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (global_id < shift) {
            sum_buffer[global_id] += sum_buffer[global_id + shift];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float normalization_factor = sum_buffer[0];

    // get probabilities
    buffer[global_id] /= normalization_factor;
}
