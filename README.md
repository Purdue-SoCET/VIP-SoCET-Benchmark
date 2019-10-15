# VIP-SoCET-Benchmark

# References


# Quick start guide
clone repository

execute run_homemade_int_inference.py

# File Descriptions
Look at Some MNIST.py:

Shows a few images from MNIST dataset

Train MNIST.py:

Used for analyzing sparsity in CNN

frozen_model.pb:

original model before quantization, i.e. floating point weights ect.

integer_inference.py:

code containing homemade integer only inferencing, adapted from:

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/internal/reference

see conv.h, fully_connected.h, common.h

https://github.com/google/gemmlowp/

see fixedpoint/fixed_point.h

make_tflite.py:

creates model.tflite, compares accuracy between model.tflite and frozen_model.pb

model.tflite:

quantized version of frozen_model.pb, i.e. uint8 weights, ect.

offline.py:

functions related to parameters calculated "offline" meaning it is calculated before running integer only inferencing

run_homemade_int_inference.py:

compares inferencing using integer_inference and using Tensorflow Lite

softmax.py:

scratch code used when trying to implement softmax





