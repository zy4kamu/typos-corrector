__kernel void initialize(__global float* input_and_hidden_buffer, __global float* state_buffer,
                         int input_size, int lstm_size) {
    int global_id = get_global_id(0);
    input_and_hidden_buffer[global_id + input_size] = 0;
    state_buffer[global_id] = 0;
}

float exponent(float value) {
    if (value > 10) {
        return 22026;
    } else if (value < -10) {
        return 0;
    }
    return exp(value);
}

__kernel void lstm_cell(__global float* input_and_hidden_buffer, __global float* state_buffer, 
                        __global float* ijfo_buffer, int input_size, int lstm_size)
{
    // unpack gates
    size_t global_id = get_global_id(0);
    float input_gate      = *(ijfo_buffer                 + global_id);
    float activation_gate = *(ijfo_buffer +     lstm_size + global_id);
    float forget_gate     = *(ijfo_buffer + 2 * lstm_size + global_id);
    float output_gate     = *(ijfo_buffer + 3 * lstm_size + global_id);

    // forget information
    __global float* state = state_buffer + global_id;
    *state = *state / (1. + exponent(-1. - forget_gate));
    
    // update information
    float input_sigma = 1. / (1. + exponent(-input_gate));
    float exp_activation = exponent(-2.0 * activation_gate);
    float tanh_activation = (1. - exp_activation) / (1. + exp_activation);
    *state += input_sigma * tanh_activation;

    // update output
    __global float* hidden = input_and_hidden_buffer + input_size + global_id;
    *hidden = *state / (1. + exponent(-output_gate));
}
