"""
Homemade implementation for inferencing using 'model.tflite' compares 'My Accuracy' with Tensorflow's implementation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import offline
import integer_inference

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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

flat_train = flat_train[..., np.newaxis]
flat_test = flat_test[..., np.newaxis]

# load TFLite file
interpreter = tf.lite.Interpreter(model_path=f'model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

# get some informations .
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inter_layer = interpreter.get_tensor_details()

tensorflow_acc = 0
my_acc = 0

for i in range(100):
    quantized_input = offline.quantize(input_details[0], flat_test[i:i+1])
    interpreter.set_tensor(input_details[0]['index'], quantized_input)

    interpreter.invoke()

    # The results are stored on 'index' of output_details
    quantized_output = interpreter.get_tensor(output_details[0]['index'])

    #print('Tensorflow results for quantized model')
    #print(offline.dequantize(output_details[0], quantized_output))

    # Hardcoded values for specific weights/biases ect. The application Netron was very helpful in understanding
    # inputs and outputs to different layers. Netron gives a good overview of what a model looks like
    weight_index = 4
    bias_index = 6
    output_index = 1
    input_index = 7
    quantized_weight = interpreter.get_tensor(inter_layer[weight_index]['index'])
    quantized_bias = interpreter.get_tensor(inter_layer[bias_index]['index'])
    quantized_correct_output = interpreter.get_tensor(inter_layer[output_index]['index'])
    weight_scale, weight_offset = inter_layer[weight_index]['quantization']
    input_scale, input_offset = inter_layer[input_index]['quantization']
    output_scale, output_offset = inter_layer[output_index]['quantization']

    M = (input_scale * weight_scale) / output_scale
    right_shift, M_0 = offline.quantize_mult_smaller_one(M)

    output_conv_arr = (integer_inference.Conv(quantized_input, input_offset, quantized_weight,
                                                             weight_offset, quantized_bias, output_offset, M_0,
                                                             right_shift, (784,16)))


    test_img = flat_test[:1]
    weight_index = 10
    bias_index = 8
    output_index = 9
    input_index = 0
    quantized_weight = interpreter.get_tensor(inter_layer[weight_index]['index'])
    quantized_bias = interpreter.get_tensor(inter_layer[bias_index]['index'])
    weight_scale, weight_offset = inter_layer[weight_index]['quantization']
    input_scale, input_offset = inter_layer[input_index]['quantization']
    output_scale, output_offset = inter_layer[output_index]['quantization']

    M = (input_scale * weight_scale) / output_scale
    right_shift, M_0 = offline.quantize_mult_smaller_one(M)

    output_conv_arr = output_conv_arr.flatten()
    output_conv_arr = output_conv_arr[np.newaxis, ...]

    output_full_conn_arr = (integer_inference.FullyConnected(output_conv_arr, input_offset, quantized_weight,
                                                             weight_offset, quantized_bias, output_offset, M_0,
                                                             right_shift, (1,128)))

    weight_index = 14
    bias_index = 12
    output_index = 11
    input_index = 9
    quantized_weight = interpreter.get_tensor(inter_layer[weight_index]['index'])
    quantized_bias = interpreter.get_tensor(inter_layer[bias_index]['index'])
    quantized_correct_output = interpreter.get_tensor(inter_layer[output_index]['index'])
    weight_scale, weight_offset = inter_layer[weight_index]['quantization']
    input_scale, input_offset = inter_layer[input_index]['quantization']
    output_scale, output_offset = inter_layer[output_index]['quantization']

    M = (input_scale * weight_scale) / output_scale
    right_shift, M_0 = offline.quantize_mult_smaller_one(M)

    output_full_conn_arr_2 = (integer_inference.FullyConnected(output_full_conn_arr, input_offset, quantized_weight,
                                                             weight_offset, quantized_bias, output_offset, M_0,
                                                             right_shift, (1,10)))
    if test_labels[i] == np.argmax(quantized_correct_output):
        tensorflow_acc += 1
    if test_labels[i] == np.argmax(output_full_conn_arr_2):
        my_acc += 1
    print('Tensorflow Working Accuracy: ', tensorflow_acc / (i + 1))
    print('My Working Accuracy: ', my_acc / (i + 1))

print('Final Tensorflow Accuracy: ', tensorflow_acc / 100)
print('Final My Accuracy: ', my_acc / 100)
