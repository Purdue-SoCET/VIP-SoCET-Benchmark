import tensorflow as tf
import offline

# load TFLite file
interpreter = tf.lite.Interpreter(model_path=f'model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inter_layer = interpreter.get_tensor_details()

weight_index = 4
bias_index = 6
output_index = 1
input_index = 7
quantized_weight_conv = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_conv = interpreter.get_tensor(inter_layer[bias_index]['index'])
# offsets only
weight_scale_conv, weight_offset_conv = inter_layer[weight_index]['quantization']
input_scale_conv, input_offset_conv = inter_layer[input_index]['quantization']
output_scale_conv, output_offset_conv = inter_layer[output_index]['quantization']
#right_shift_conv and M_0_conv
M_conv = (input_scale_conv * weight_scale_conv) / output_scale_conv
right_shift_conv, M_0_conv = offline.quantize_mult_smaller_one(M_conv)

# hidden dense layer offline parameters
weight_index = 10
bias_index = 8
output_index = 9
input_index = 0
quantized_weight_dense = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_dense = interpreter.get_tensor(inter_layer[bias_index]['index'])
weight_scale_dense, weight_offset_dense = inter_layer[weight_index]['quantization']
input_scale_dense, input_offset_dense = inter_layer[input_index]['quantization']
output_scale_dense, output_offset_dense = inter_layer[output_index]['quantization']
M_dense = (input_scale_dense * weight_scale_dense) / output_scale_dense
right_shift_dense, M_0_dense = offline.quantize_mult_smaller_one(M_dense)

# prediction layer offline parameters
weight_index = 14
bias_index = 12
output_index = 11
input_index = 9
quantized_weight_pred = interpreter.get_tensor(inter_layer[weight_index]['index'])
quantized_bias_pred = interpreter.get_tensor(inter_layer[bias_index]['index'])
weight_scale_pred, weight_offset_pred = inter_layer[weight_index]['quantization']
input_scale_pred, input_offset_pred = inter_layer[input_index]['quantization']
output_scale_pred, output_offset_pred = inter_layer[output_index]['quantization']
M_pred = (input_scale_pred * weight_scale_pred) / output_scale_pred
right_shift_pred, M_0_pred = offline.quantize_mult_smaller_one(M_pred)

pass