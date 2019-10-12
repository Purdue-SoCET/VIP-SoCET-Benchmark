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

def exp_on_interval_between_negative_one_quarter_and_0_excl(x):
    # x between [-1/4, 0)
    # shift to [-1/8, 1/8)
    # floating point representation:
    # bit index range: function
    # 31: sign 30-26: whole number 25-0: float
    # i.e. decimal place between bits 25 and 26 (starting at index 0)

    # Taylor Series Expansion
    x = x + (1 << 23)
    x_2 = x * x



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

output_full_conn_arr_2 = np.zeros(shape=(1, 10), dtype=np.uint8)
input_scale, input_offset = inter_layer[0]['quantization']
output_scale, output_offset = inter_layer[2]['quantization']
M = input_scale / output_scale
left_shift, M_0 = quantize_mult_greater_one(M)
quantized_correct_output = interpreter.get_tensor(inter_layer[2]['index'])
max_in_row = max(output_full_conn_arr[0])
diff_min = -255

sum_of_exps = 0
temp_arr = []
# Sum of Exps
for i in range(10):
    input_val = np.int32(output_full_conn_arr[0][i])
    input_diff = np.int32(input_val - max_in_row)
    input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, M_0, left_shift)

    temp_arr.append(input_diff_rescaled)

pass
    # if input_diff >= diff_min:
    #     input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, M_0, left_shift)
    #     scaled_diff_f8 = FixedPointScaledDiff_Raw(input_diff_rescaled)
    #     sum_of_exps += Rescale(exp_on_negative_vales(scaled_diff_f8))

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
