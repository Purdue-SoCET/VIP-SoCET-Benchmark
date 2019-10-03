import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from keras_quantizer import quantize_mult_smaller_one

"""
https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=jWp9_I06ZjDo
"""

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# def build_keras_model():
#     return keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(128, activation='relu'),
#         # keras.layers.BatchNormalization(),
#         keras.layers.Dense(10, activation='softmax')
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
#     train_model.fit(train_images, train_labels, epochs=5)
#
#     # save graph and checkpoints
#     saver = tf.train.Saver()
#     saver.save(train_sess, 'checkpoints')
#
# with train_graph.as_default():
#     print('sample result of original model')
#     print(train_model.predict(test_images[:1]))
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


def SatRoundDoublingHighMul(a, b):
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


def MultByQuantMultSmallThanOne(x, quant_mul, right_shift):
    return RoundDividByPOT(SatRoundDoublingHighMul(x, quant_mul), right_shift)

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

quantized_input = quantize(input_details[0], test_images[:1])
interpreter.set_tensor(input_details[0]['index'], quantized_input)

interpreter.invoke()

# The results are stored on 'index' of output_details
quantized_output = interpreter.get_tensor(output_details[0]['index'])
full_quant_out = interpreter.get_tensor(inter_layer[6]['index'])


print('sample result of quantized model')
print(dequantize(output_details[0], quantized_output))


test_img = test_images[:1]
quantized_input = quantized_input.flatten()
output_full_conn_arr = np.zeros(shape=(1, 128), dtype=np.uint8)
quantized_weight = interpreter.get_tensor(inter_layer[2]['index'])
quantized_bias = interpreter.get_tensor(inter_layer[0]['index'])
quantized_correct_output = interpreter.get_tensor(inter_layer[1]['index'])
weight_scale, weight_offset = inter_layer[2]['quantization']
input_scale, input_offset = inter_layer[7]['quantization']
output_scale, output_offset = inter_layer[1]['quantization']

M = (input_scale * weight_scale) / output_scale
right_shift, M_0 = quantize_mult_smaller_one(M)
# int only attempt

x = quantized_weight[0, :]
# first fully connected layer int
for i in range(128):
    acc = np.int32(0)
    for j in range(784):
        input_val = np.int32(quantized_input[j])
        weight_val = np.int32(quantized_weight[i][j])
        acc += (input_val - input_offset) * (weight_val - weight_offset)
    acc += quantized_bias[i]
    acc = MultByQuantMultSmallThanOne(acc, M_0, right_shift)
    acc += output_offset  # activation offset
    acc = np.max([acc, np.int32(0)])
    acc = np.min([acc, np.int32(255)])
    output_full_conn_arr[0][i] = acc

print(output_full_conn_arr)
print(quantized_correct_output)


# second fully connected layer
output_full_conn_arr_2 = np.zeros(shape=(1, 10), dtype=np.uint8)
quantized_weight = interpreter.get_tensor(inter_layer[6]['index'])
quantized_bias = interpreter.get_tensor(inter_layer[4]['index'])
quantized_correct_output = interpreter.get_tensor(inter_layer[3]['index'])
weight_scale, weight_offset = inter_layer[6]['quantization']
input_scale, input_offset = inter_layer[1]['quantization']
output_scale, output_offset = inter_layer[3]['quantization']

M = (input_scale * weight_scale) / output_scale
right_shift, M_0 = quantize_mult_smaller_one(M)
for i in range(10):
    acc = np.int32(0)
    for j in range(128):
        input_val = np.int32(output_full_conn_arr[0][j])
        weight_val = np.int32(quantized_weight[i][j])
        acc += (input_val - input_offset) * (weight_val - weight_offset)
    acc += quantized_bias[i]
    acc = MultByQuantMultSmallThanOne(acc, M_0, right_shift)
    acc += output_offset  # activation offset
    acc = np.max([acc, np.int32(0)])
    acc = np.min([acc, np.int32(255)])
    output_full_conn_arr_2[0][i] = acc

# softmax layer
output_full_conn_arr_3 = np.zeros(shape=(1, 10), dtype=np.uint8)
input_scale, input_offset = inter_layer[3]['quantization']
output_scale, output_offset = inter_layer[5]['quantization']

M = input_scale / output_scale
right_shift, M_0 = quantize_mult_smaller_one(M)

for i in range(10):
    acc = np.int32(0)
    input_val = np.int32(output_full_conn_arr_2[0][i])
    acc = input_val - input_offset
    acc = MultByQuantMultSmallThanOne(acc, M_0, right_shift)
    acc += output_offset  # activation offset
    acc = np.max([acc, 0])
    acc = np.min([acc, 255])
    output_full_conn_arr_3[0][i] = np.uint8(acc)

print(output_full_conn_arr_3)


