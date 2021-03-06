#ifndef INTEGER_INFERENCE_H
#define INTEGER_INFERENCE_H
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "img.h"
#include "model.h"
#include "dense.h"
#include "pred.h"

//USED
int32_t maxOf(int32_t a, int32_t b) {
	if (a > b) return a;
	else return b;
}

int32_t minOf(int32_t a, int32_t b) {
	if (a < b) return a;
	else return b;
}

int32_t RoundingDivideByPOT(int32_t x, int exponent) { //DONE
	if ((exponent < 0) | (exponent > 31)) perror("RoundingDivideByPOT: Inputs Incorrect\n"); //CHECK
	int32_t mask = (int32_t)((1 << exponent) - 1);
	int32_t zero = (int32_t)(0);
	int32_t one = (int32_t)(1);
	int32_t remainder = x & mask;

	int32_t maskiflessthan = x;
	if (x < zero) maskiflessthan &= zero;

	int32_t threshold = (mask >> 1) + (maskiflessthan & one);
	int32_t maskifgreaterthan = remainder;
	if (remainder > threshold) maskifgreaterthan &= threshold;

	return (x >> exponent) + (maskifgreaterthan & one);
}

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) { //DONE
	bool overflow = (a == b) & (a == 0-2147483648);
	if (overflow) return (int32_t)(2147483647);

	int64_t a_64 = (int64_t) a;
	int64_t b_64 = (int64_t) b;
	int64_t ab_64 = a_64 * b_64;
	int64_t nudge;

	if (ab_64 >= 0) nudge = (int32_t)(1 << 30);
	else nudge = (int32_t)(1 - (1 << 30));

	int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (int64_t)((int64_t)(1) << 31));
	return ab_x2_high32;
}

int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quant_mul, int left_shift) { //DONE
	return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul);
}

int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x, int32_t quant_mul, int right_shift) { //DONE
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift);
}

const int nCols = 14;
const int nRows = 14;
const int imgSize = nCols * nRows;
const int nFilters = 8;
const int kernel_shape = 3;
const int convINshape[3] = { 1, imgSize, 1 };
const int convWshape[4] = { 1, 1, kernel_shape, nFilters };
const int convBshape[1] = { nFilters };
const int convOUTshape[2] = { imgSize, nFilters }; //was previously int* output_shape in Conv()


void Conv(uint8_t output_conv_arr[imgSize][nFilters]) {
	int rows = convOUTshape[1];
	int cols = convOUTshape[0];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			if (j + 1 <= cols - 1) {
				for (int k = 0; k < kernel_shape; k++) {
					input_val = (int32_t)(img[0][j - 1 + k][0]);
					weight_val = (int32_t)(quantized_weight_conv[0][0][k][i]);
					acc += (input_val - input_offset_conv) * (weight_val - weight_offset_conv);
				}
			}
			acc += quantized_bias_conv[i];
			acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_conv, right_shift_conv);
			acc += output_offset_conv; // activation offset
			acc = maxOf(acc, (int32_t)(0));
			acc = minOf(acc, (int32_t)(255));
			output_conv_arr[j][i] = (uint8_t)(acc); //RETURN ARRAY
			//printf("[%d][%d]: %d\n", j, i, output_conv_arr[j][i]);
		}
	}
}

void flattenConv(uint8_t arrIN[imgSize][nFilters], uint8_t arrOUT[imgSize*nFilters]) {
	for (int i = 0; i < imgSize; i++) {
		for (int j = 0; j < nFilters; j++) {
			arrOUT[i*nFilters + j] = arrIN[i][j];
			//printf("[%d]: %d\n", i*nFilters + j, arrOUT[i*nFilters + j]);
		}
	}
}


const int n_dense_nodes = 16;
const int fcdINshape[2] = { 1, imgSize*nFilters};
const int fdcWshape[2] = { n_dense_nodes, imgSize*nFilters };
const int fcdBshape[1] = { n_dense_nodes };
const int fcdOUTshape[2] = { 1, n_dense_nodes };
void FullyConnectedDense(uint8_t quantized_inputs[1][imgSize*nFilters], uint8_t output_full_conn_dense_arr[1][n_dense_nodes]) {
	int rows = fdcWshape[0];
	int cols = fdcWshape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			input_val = (int32_t)(quantized_inputs[0][j]);
			weight_val = (int32_t)(quantized_weight_dense[i][j]);
			acc += (input_val - input_offset_dense) * (weight_val - weight_offset_dense);
		}
		acc += quantized_bias_dense[i];
		acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_dense, right_shift_dense);
		acc += output_offset_dense;  // activation offset
		acc = maxOf(acc, (int32_t)(0));
		acc = minOf(acc, (int32_t)(255));
		output_full_conn_dense_arr[0][i] = (uint8_t)(acc);
		printf("[%d]: %d\n", i, output_full_conn_dense_arr[0][i]);
	}
}

const int n_pred_nodes = 10;
const int fcpINshape[2] = { 1, n_dense_nodes };
const int fcpWshape[2] = { n_pred_nodes, n_dense_nodes };
const int fcpBshape[1] = { n_pred_nodes };
const int fcpOUTshape[2] = { 1, n_pred_nodes };
void FullyConnectedPred(uint8_t quantized_inputs[1][n_dense_nodes], uint8_t output_full_conn_pred_arr[1][n_pred_nodes]) {
	int rows = fcpWshape[0];
	int cols = fcpWshape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			input_val = (int32_t)(quantized_inputs[0][j]);
			weight_val = (int32_t)(quantized_weight_pred[i][j]);
			acc += (input_val - input_offset_pred) * (weight_val - weight_offset_pred);
		}
		acc += quantized_bias_pred[i];
		acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0_pred, right_shift_pred);
		acc += output_offset_pred;  // activation offset
		acc = maxOf(acc, (int32_t)(0));
		acc = minOf(acc, (int32_t)(255));
		output_full_conn_pred_arr[0][i] = (uint8_t)(acc);
		printf("[%d]: %d\n", i, output_full_conn_pred_arr[0][i]);
	}
}

#endif
