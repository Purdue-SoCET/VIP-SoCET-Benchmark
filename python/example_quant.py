import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from keras_quantizer import quantize_mult_smaller_one, quantize_mult_greater_one
from pprint import pprint as pp


"""
https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=jWp9_I06ZjDo
"""

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(test_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

flat_train = []
flat_test = []

for i, img in enumerate(train_images):
    flat_train.append(img.flatten())
flat_train = np.asarray(flat_train)

for i, img in enumerate(test_images):
    flat_test.append(img.flatten())
flat_test = np.asarray(flat_test)


# def build_keras_model():
#     return keras.Sequential([
#         #keras.layers.Flatten(input_shape=(28, 28)),
#         # keras.layers.BatchNormalization(),
#         keras.layers.Dense(10, activation='softmax', input_shape=(784,))
#     ])
#
#
# train_graph = tf.Graph()
# train_sess = tf.Session(graph=train_graph)
#
# keras.backend.set_session(train_sess)
# with train_graph.as_default():
#     train_model = build_keras_model()
#
#     tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
#     train_sess.run(tf.global_variables_initializer())
#
#     train_model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     train_model.fit(flat_train, train_labels, epochs=5)
#
#     # save graph and checkpoints
#     saver = tf.train.Saver()
#     saver.save(train_sess, 'checkpoints')
#
# with train_graph.as_default():
#     print('sample result of original model')
#     print(train_model.predict(flat_test[:1]))
#
# # eval
# eval_graph = tf.Graph()
# eval_sess = tf.Session(graph=eval_graph)
#
# keras.backend.set_session(eval_sess)
#
# with eval_graph.as_default():
#     keras.backend.set_learning_phase(0)
#     eval_model = build_keras_model()
#     tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
#     eval_graph_def = eval_graph.as_graph_def()
#     saver = tf.train.Saver()
#     saver.restore(eval_sess, 'checkpoints')
#
#     frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#         eval_sess,
#         eval_graph_def,
#         [eval_model.output.op.name]
#     )
#
#     with open('frozen_model.pb', 'wb') as f:
#         f.write(frozen_graph_def.SerializeToString())
#


def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data / a + b).astype(dtype).reshape(shape)


def dequantize(detail, data):
    a, b = detail['quantization']

    return (data - b) * a



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

"""
inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}
"""

"""
inline int32 MultiplyByQuantizedMultiplierGreaterThanOne(
    int32 x, int32 quantized_multiplier, int left_shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                           quantized_multiplier);
}
"""


def MultiplyByQuantizedMultiplierGreaterThanOne(x, quant_mul, left_shift):
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quant_mul)


# https://github.com/tensorflow/tensorflow/blob/4952f981be07b8bf508f8226f83c10cdafa3f0c4/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h
def MultiplyByQuantizedMultiplierSmallerThanOne(x, quant_mul, right_shift):
    return RoundDividByPOT(SaturatingRoundingDoublingHighMul(x, quant_mul), right_shift)

"""
    inline int32 MultiplyByQuantizedMultiplierSmallerThanOne(
        int32 x, int32 quantized_multiplier, int right_shift) {
      using gemmlowp::RoundingDivideByPOT;
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      return RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(x, quantized_multiplier), right_shift);
    }
"""


def RoundDividByPOT(x, exponent):
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

"""
// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
template <typename IntegerType, typename ExponentType>
inline IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent) {
  assert(exponent >= 0);
  assert(exponent <= 31);
  const IntegerType mask = Dup<IntegerType>((1ll << exponent) - 1);
  const IntegerType zero = Dup<IntegerType>(0);
  const IntegerType one = Dup<IntegerType>(1);
  const IntegerType remainder = BitAnd(x, mask);
  const IntegerType threshold =
      Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
  return Add(ShiftRight(x, exponent),
             BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}
"""

def SaturatingRoundingMultiplyByPOT(x, exponent):
    return RoundDividByPOT(x, -exponent)

def SaturatingAdd(a, b):
    a32 = np.int32(a)
    b32 = np.int32(b)
    sum = np.int64(a32 + b32)
    sum = np.max([-32768, sum])
    sum = np.min([32767, sum])
    return np.int16(sum)

def exp_on_interval_between_negative_one_quarter_and_0_excl(x, num_int_bits):
    if x < float_to_q(-0.25, 5):
        raise ValueError('Must be > -0.25')
    elif x > 0:
        raise ValueError('Must be < 0')
    mask = (1 << (32 - num_int_bits - 1)) - 1
    mask_shift = np.int32((x & mask) << num_int_bits)
    rescaled_x = np.int32(mask_shift | (1 << 31))
    scaled_x = np.int32(rescaled_x + (1 << 28))
    x2 = SaturatingRoundingDoublingHighMul(scaled_x, scaled_x)
    x3 = SaturatingRoundingDoublingHighMul(x2, scaled_x)
    x4 = SaturatingRoundingDoublingHighMul(x2, x2)
    x4_over_4 = SaturatingRoundingMultiplyByPOT(x4, -2)
    constant_1_over_3 = float_to_q(1/3, 0)
    constant_term = float_to_q(np.exp(-1.0 / 8.0), 0)
    x4_over_24_plus_x3_over_6_plus_x2_over_2 = SaturatingRoundingMultiplyByPOT(SaturatingRoundingDoublingHighMul(x4_over_4 + x3, constant_1_over_3) + x2, -1)
    x4_over_24_plus_x3_over_6_plus_x2_over_2_mul_constant_term = SaturatingRoundingDoublingHighMul(constant_term, x4_over_24_plus_x3_over_6_plus_x2_over_2)
    result = SaturatingAdd(constant_term, x4_over_24_plus_x3_over_6_plus_x2_over_2_mul_constant_term)

    if result < 0:
        raise ValueError('Negative Output')
    return result


def change_to_float(int_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return int_val * (2 ** -n)


def float_to_q(float_val, num_int_bits):
    n = 32 - num_int_bits - 1
    return np.int32(round(float_val * (2 ** n)))


def CountLeadingZeros(x):
    count = 0
    for i in range(32):
        if (x >> (31 - i)) == 0:  # ignore sign? i think so
            count += 1
        else:
            break
    return count


def RoundingHalfSum(a, b):
    a64 = np.int64(a)
    b64 = np.int64(b)
    sum = a64 + b64
    if sum >= 0:
        sign = 1
    else:
        sign = -1
    return np.int32((sum + sign) / 2)


# returns 1 / (1 + x) for x (0, 1)
def one_over_one_plus_x_for_x_in_0_1(a, num_int_bits):
    if (change_to_float(a, num_int_bits) > 1) | (change_to_float(a, num_int_bits) < 0):
        raise ValueError('input not between 0 and 1')
    half_denominator = RoundingHalfSum(a, 2147483647)
    constant_48_over_17 = float_to_q(48.0/17.0, 2)
    constant_neg_32_over_17 = float_to_q(-32.0/17.0, 2)
    constant_one = float_to_q(1.0, 2)
    half_denominator_mul_constant_neg_32_over_17 = SaturatingRoundingDoublingHighMul(half_denominator, constant_neg_32_over_17)
    x = constant_48_over_17 + half_denominator_mul_constant_neg_32_over_17
    for i in range(3):
        half_denominator_times_x = SaturatingRoundingDoublingHighMul(half_denominator, x)
        one_minus_half_denominator_times_x = constant_one - half_denominator_times_x
        x = x + SaturatingRoundingDoublingHighMul(x, one_minus_half_denominator_times_x)
    return np.int32(x << 1)


def exp_on_negative_values(a, num_int_bits):
    # change to 0 bit int rep
    mask = (1 << 24) - 1
    one_quarter = 1 << 24
    a_mod_quarter_minus_one_quarter = (input_diff_rescaled & mask) - one_quarter
    result = exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_quarter_minus_one_quarter, 5)
    remainder = a_mod_quarter_minus_one_quarter - a
    result = SelectUsingMask(MaskIfZero(a), (1 << 31) - 1, result)
    return result
    # constant_one = 2147483647
    # constant_half = float_to_q(0.5, num_int_bits)
    # constant_1_over_6 = float_to_q(1 / 6, num_int_bits)
    # constant_1_over_24 = float_to_q(1 / 24, num_int_bits)
    # one_plus_x = constant_one + x
    # x2 = FixedPointMul(x, x, num_int_bits)
    # x2_over_2 = FixedPointMul(x2, constant_half, num_int_bits)
    # x3 = FixedPointMul(x2, x, num_int_bits)
    # x3_over_6 = FixedPointMul(x3, constant_1_over_6, num_int_bits)
    # x4 = FixedPointMul(x2, x2, num_int_bits)
    # x4_over_24 = FixedPointMul(x4, constant_1_over_24, num_int_bits)
    # return one_plus_x + x2_over_2 + x3_over_6 + x4_over_24


def MaskIfNonZero(a):
    if a:
        return ~np.int32(0)
    else:
        return np.int32(0)


def MaskIfZero(a):
    return MaskIfNonZero(~a)


def SelectUsingMask(if_mask, then_val, else_val):
    return (if_mask & then_val) ^ ((~if_mask) & else_val)


# load TFLite file
interpreter = tf.lite.Interpreter(model_path=f'model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

# get some informations .
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inter_layer = interpreter.get_tensor_details()

print(input_details)
print(output_details)

quantized_input = quantize(input_details[0], flat_test[:1])
interpreter.set_tensor(input_details[0]['index'], quantized_input)

interpreter.invoke()

# The results are stored on 'index' of output_details
quantized_output = interpreter.get_tensor(output_details[0]['index'])
full_quant_out = interpreter.get_tensor(inter_layer[2]['index'])


print('sample result of quantized model')
print(dequantize(output_details[0], quantized_output))


test_img = flat_test[:1]
output_full_conn_arr = np.zeros(shape=(1, 10), dtype=np.uint8)
quantized_weight = interpreter.get_tensor(inter_layer[3]['index'])
quantized_bias = interpreter.get_tensor(inter_layer[1]['index'])
quantized_correct_output = interpreter.get_tensor(inter_layer[0]['index'])
weight_scale, weight_offset = inter_layer[3]['quantization']
input_scale, input_offset = inter_layer[4]['quantization']
output_scale, output_offset = inter_layer[0]['quantization']

M = (input_scale * weight_scale) / output_scale
right_shift, M_0 = quantize_mult_smaller_one(M)
# int only attempt

# first fully connected layer int
for i in range(10):
    acc = np.int32(0)
    for j in range(784):
        input_val = np.int32(quantized_input[0][j])
        weight_val = np.int32(quantized_weight[i][j])
        acc += (input_val - input_offset) * (weight_val - weight_offset)
    acc += quantized_bias[i]
    acc = MultiplyByQuantizedMultiplierSmallerThanOne(acc, M_0, right_shift)
    acc += output_offset  # activation offset
    acc = np.max([acc, np.int32(0)])
    acc = np.min([acc, np.int32(255)])
    output_full_conn_arr[0][i] = np.uint8(acc)

pp(output_full_conn_arr)
pp(quantized_correct_output)
pp(dequantize(inter_layer[0], quantized_correct_output))

output_full_conn_arr_2 = np.zeros(shape=(1, 10), dtype=np.uint8)
input_scale, input_offset = inter_layer[0]['quantization']
output_scale, output_offset = inter_layer[2]['quantization']
M = input_scale / output_scale
left_shift, M_0 = quantize_mult_greater_one(M)
quantized_correct_output = interpreter.get_tensor(inter_layer[2]['index'])
max_in_row = max(output_full_conn_arr[0])
diff_min = -25

# input_scale_fixed_point = float_to_q(input_scale, 0)
# scaled_val = np.int32(output_full_conn_arr[0][8]) - input_offset
# float_input_val = (np.int64(input_scale_fixed_point) * scaled_val) >> 12

sum_of_exps = 0
temp_arr = []
# Sum of Exps
for i in range(10):
    input_val = np.int32(output_full_conn_arr[0][i])
    input_diff = input_val - max_in_row
    if input_diff >= -32:
        input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, M_0, left_shift)
        result = exp_on_negative_values(input_diff_rescaled, 5)
        mask = (1 << (32 - 0 - 1)) - 1
        mask_shift = np.int32((result & mask) >> 12)
        rescaled_x = np.int32(mask_shift | (0 << 31))  # stay pos
        sum_of_exps += rescaled_x


# # sum_of_exps = float_to_q(1.25, 12)
headroom_plus_one = CountLeadingZeros(np.uint32(sum_of_exps))
num_bits_over_unit = 12 - headroom_plus_one
shifted_sum_minus_one = np.int32((np.uint32(sum_of_exps) << headroom_plus_one) - (np.uint32(1) << 31))
shifted_scale = one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one, 0)
#
for i in range(10):
    input_val = np.int32(output_full_conn_arr[0][i])
    input_diff = np.int32(input_val - max_in_row)
    if input_diff >= -32:
        input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, M_0, left_shift)
        result = exp_on_negative_values(input_diff_rescaled, 5)
        scaled_result = SaturatingRoundingDoublingHighMul(shifted_scale, result)
        unsat_output = RoundDividByPOT(scaled_result, num_bits_over_unit + 31 - 8)
        unsat_output = np.min([unsat_output, np.int32(255)])
        unsat_output = np.max([unsat_output, np.int32(0)])
        output_full_conn_arr_2[0][i] = np.uint8(unsat_output)
    else:
        output_full_conn_arr_2[0][i] = np.uint8(0)

pp(output_full_conn_arr_2)
pp(quantized_correct_output)

# fixed_sum_of_exps = FixedPointAcc_Raw(sum_of_exps)
# headroom_plus_one = CountLeadingZeros(fixed_sum_of_exps)
# num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one
# shifted_sum_minus_one = np.int32((np.int32(fixed_sum_of_exps) << headroom_plus_one) - (np.int32(1) << 31))
# shifted_scale = one_over_one_plus_x_for_x_in_0_1(FixedPoint0_Raw(shifted_sum_minus_one))
#
# for i in range(10):
#     input_val = np.int32(output_full_conn_arr[0][i])
#     input_diff = np.int32(input_val - max_in_row)
#     if input_diff >= diff_min:
#         input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, M_0, left_shift)
#         scaled_diff_f8 = FixedPointScaledDiff_Raw(input_diff_rescaled)
#         sum_of_exps += Rescale(exp_on_negative_vales(scaled_diff_f8))
#         exp_in_0 = exp_on_negative_vales(scaled_diff_f8)
#         unsat_output = RoundDividByPOT(FixedPoint0_Raw(shifted_scale * exp_in_0), num_bits_over_unit + 31 - 8)
#         unsat_output = np.max([unsat_output, np.int32(0)])
#         unsat_output = np.min([unsat_output, np.int32(255)])
#         output_full_conn_arr_2[0][i] = np.uint8(unsat_output)
#     else:
#         output_full_conn_arr_2[0][i] = 0
