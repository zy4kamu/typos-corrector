__kernel void reset(__global float* input_and_hidden_buffer, __global float* state_buffer,
                    int input_size, int lstm_size) {
    int global_id = get_global_id(0);
    input_and_hidden_buffer[global_id + input_size] = 0;
    state_buffer[global_id] = 0;
}

__kernel void set_input(__global float* input_and_hidden_buffer, int one_hot_index) {
    int global_id = get_global_id(0);
    if (global_id == one_hot_index) {
        input_and_hidden_buffer[global_id] = 1;
    } else {
        input_and_hidden_buffer[global_id] = 0;
    }
}

float sigmoid(float value) {
    return 1. / (1. + exp(-value));
}

float hyperbolic_tangent(float value) {
    float exp_activation = exp(-2.0 * value);
    return (1. - exp_activation) / (1. + exp_activation);
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
    *state *= sigmoid(1. + forget_gate);

    // update information
    *state += sigmoid(input_gate) * hyperbolic_tangent(activation_gate);

    // update output
    __global float* hidden = input_and_hidden_buffer + input_size + global_id;
    *hidden = hyperbolic_tangent(*state) * sigmoid(output_gate);
}
