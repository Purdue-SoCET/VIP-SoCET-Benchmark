// VIPmodel.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdint.h>
#include "integer_inference.h"

int main()
{
	uint8_t output_conv_arr[imgSize][nFilters];
	Conv(output_conv_arr);
	uint8_t output_conv_arr_flat[1][imgSize*nFilters];
	flattenConv(output_conv_arr, output_conv_arr_flat[0]);
	
	uint8_t output_full_conn_dense_arr[1][n_dense_nodes];
	FullyConnectedDense(output_conv_arr_flat, output_full_conn_dense_arr);

	uint8_t output_full_conn_pred_arr[1][n_pred_nodes];
	FullyConnectedPred(output_full_conn_dense_arr, output_full_conn_pred_arr);

	//
	//	uint8_t output_full_conn_arr_2 [1][10];
	//	int output_shape_pred[2] = {1, 10};
	//	int input_shape_pred[2] = {1, 16};
	//
	//	FullyConnected(&output_full_conn_arr, input_offset_pred, &input_shape_pred,
	//				   &quantized_weight_pred, weight_offset_pred,
	//				   &quantized_bias_pred, output_offset_pred,
	//				   M_0_pred, right_shift_pred,
	//				   &output_full_conn_arr_2, &output_shape_pred);
	//
	//	for (int i = 0; i < 10; i++) {
	//		printf("%d", output_full_conn_arr_2[i]);
	//	}
	//	printf("%d", lbl);
    return 0;
}

