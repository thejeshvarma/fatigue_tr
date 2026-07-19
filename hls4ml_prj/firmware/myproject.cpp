#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[10],
    result_t layer9_out[3]
) {

    // hls-fpga-machine-learning insert IO
    // AXI-Lite interface for PS access (PYNQ / Vitis bare-metal)
    #pragma HLS INTERFACE s_axilite port=input_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=layer9_out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS ARRAY_PARTITION variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<dense_weight_t, 160>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 128>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 8>(b5, "b5.txt");
        nnet::load_weights_from_txt<dense_2_weight_t, 24>(w8, "w8.txt");
        nnet::load_weights_from_txt<dense_2_bias_t, 3>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    dense_result_t layer2_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer4_t layer4_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    dense_1_result_t layer5_out[8];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer7_t layer7_out[8];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    dense_2_result_t layer8_out[3];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    nnet::dense<input_t, dense_result_t, config2>(input_1, layer2_out, w2, b2); // dense

    nnet::relu<dense_result_t, layer4_t, ReLU_config4>(layer2_out, layer4_out); // re_lu

    nnet::dense<layer4_t, dense_1_result_t, config5>(layer4_out, layer5_out, w5, b5); // dense_1

    nnet::relu<dense_1_result_t, layer7_t, ReLU_config7>(layer5_out, layer7_out); // re_lu_1

    nnet::dense<layer7_t, dense_2_result_t, config8>(layer7_out, layer8_out, w8, b8); // dense_2

    nnet::softmax<dense_2_result_t, result_t, softmax_config9>(layer8_out, layer9_out); // dense_2_softmax

}

