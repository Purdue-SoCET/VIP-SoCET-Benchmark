import numpy as np


"""
Code Implemented
https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
    - SaturatingRoundingDoublingHighMul
    - RoundingDivideByPOT
    - exp_on_interval_between_negative_one_quarter_and_0_excl
    - GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT
    - RescaleConstantInitializer
    - CheckedFixedPointConstant
https://github.com/tensorflow/tensorflow/blob/4952f981be07b8bf508f8226f83c10cdafa3f0c4/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h
    - MultiplyByQuantizedMultiplierGreaterThanOne
    - MultiplyByQuantizedMultiplierSmallerThanOne
"""


def SaturatingRoundingDoublingHighMul(a, b):
    overflow = (a == b) & (a == -2147483648)
    a_64 = np.int64(a)
    b_64 = np.int64(b)

    ab_64 = a_64 * b_64
    if ab_64 >= 0:
        nudge = np.int32(1 << 30)
    else:
        nudge = np.int32(1 - (1 << 30))
    ab_x2_high32 = np.int32((ab_64 + nudge) / (np.int64(1 << 31)))

    if overflow:
        return np.int32(2147483647)
    else:
        return ab_x2_high32


def MultiplyByQuantizedMultiplierGreaterThanOne(x, quant_mul, left_shift):
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul)


def MultiplyByQuantizedMultiplierSmallerThanOne(x, quant_mul, right_shift):
    return RoundDividByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift)


def RoundingDivideByPOT(x, exponent):
    """
    Correctly-rounded-to-nearest division by a power-of-two.
    Also known as a rounding arithmetic right shift.
    """
    if (exponent < 0) | (exponent > 31):
        raise ValueError('Inputs incorrect')
    mask = np.int32((1 << exponent) - 1)
    zero = np.int32(0)
    one = np.int32(1)
    remainder = x & mask
    if x < zero:
        maskiflessthan = x & zero
    else:
        maskiflessthan = x

    threshold = (mask >> 1) + (maskiflessthan & one)

    if remainder > threshold:
        maskifgreaterthan = remainder & threshold
    else:
        maskifgreaterthan = remainder

    return (x >> exponent) + (maskifgreaterthan & one)


def exp_on_negative_values(neg_val):
    input_int_bits = 5
    input_frac_bits = 32 - (input_int_bits + 1)
    output_int_bits = 0
    output_frac_bits = 32 - (output_int_bits + 1)
    input_one_quarter = 1 << 24
    mask = input_one_quarter - 1


def exp_on_interval_between_negative_one_quarter_and_0_excl(x):
    pass


def GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(type, int32_val, double_value):
    pass


def RescaleConstantInitializer(int32_value, ScalarTypeBits):
    return RoundingDivideByPOT(int32_value, 32 - ScalarTypeBits)

def CheckedFixedPointConstant()
    pass