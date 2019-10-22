#ifndef INTEGER_INFERENCE_H
#define INTEGER_INFERENCE_H
#include <stdint.h>
#include "stdafx.h" //CHECK

//USED
int32_t maxOf(int32_t a, int32_t b) {
	if (a > b) return a;
	else return b;
}

int32_t minOf(int32_t a, int32_t b) {
	if (a < b) return a;
	else return b;
}

int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quant_mul, int left_shift) { //DONE
	return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul);
}

int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x, int32_t quant_mul, int right_shift) { //DONE
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift);
}

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) { //DONE
	bool overflow = (a == b) & (a == -2147483648);
	if (overflow) return (int32_t)(2147483647);

	int64_t a_64 = (int64_t) a;
	int64_t b_64 = (int64_t) b;
	int64_t ab_64 = a_64 * b_64;
	int32_t nudge;

	if (ab_64 >= 0) nudge = (int32_t)(1 << 30);
	else nudge = (int32_t)(1 - (1 << 30));

	int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / (int64_t)(1 << 31));
	return ab_x2_high32;
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

const int nCols = 28;
const int nRows = 28;
const int nFilters = 16;
const int imgSize = nCols * nRows;
const int convINshape[3] = { 1, imgSize, 1 };
const int convWshape[4] = { 1, 1, 3, nFilters };
const int convBshape[1] = { nFilters };
const int convOUTshape[2] = { imgSize, nFilters }; //was previously int* output_shape in Conv()
void Conv(uint8_t*** quantized_inputs, uint8_t input_offset,
			   uint8_t**** quantized_weights, uint8_t weight_offset,
			   int32_t* quantized_bias, uint8_t output_offset,
			   int32_t M_0, int right_shift, 
			   uint8_t** output_conv_arr) { //THE OUTPUT ARRAY NEEDS TO BE = { 0 } AT START
	int kernel_shape = convWshape[2];
	int rows = convWshape[3];
	int cols = convINshape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			if (j + 1 <= cols - 1) {
				for (int k = 0; k < kernel_shape; k++) {
					input_val = (int32_t)(quantized_inputs[0, j - 1 + k, 0]);
					weight_offset = (int32_t)(quantized_weights[0, 0, k, i]);
					acc += (input_val - input_offset) * (weight_val - weight_offset);
				}
			}
			acc += quantized_bias[i];
			acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0, right_shift);
			acc += output_offset; // activation offset
			acc = maxOf(acc, (int32_t)(0));
			acc = minOf(acc, (int32_t)(255));
			output_conv_arr[j][i] = (uint8_t)(acc); //RETURN ARRAY
		}
	}
}


void FullyConnected(uint8_t** quantized_inputs, uint8_t input_offset, int* input_shape,
				    uint8_t** quantized_weights, uint8_t weight_offset,
			   	    int32_t* quantized_bias, uint8_t output_offset,
				    int32_t M_0, int right_shift,
					uint8_t** output_full_conn_arr, int* output_shape) { //THE OUTPUT ARRAY NEEDS TO BE = { 0 } AT START
	int rows = output_shape[1];
	int cols = input_shape[1];

	int32_t acc;
	int32_t input_val;
	int32_t weight_val;
	for (int i = 0; i < rows; i++) {
		acc = 0;
		for (int j = 0; j < cols; j++) {
			input_val = (int32_t)(quantized_inputs[0][j]);
			weight_val = (int32_t)(quantized_weights[i][j]);
			acc += (input_val - input_offset) * (weight_val - weight_offset);
		}
		acc += quantized_bias[i];
		acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0, right_shift);
		acc += output_offset;  // activation offset
		acc = maxOf(acc, (int32_t)(0));
		acc = minOf(acc, (int32_t)(255));
		output_full_conn_arr[0][i] = (uint8_t)(acc);
	}
}

#endif