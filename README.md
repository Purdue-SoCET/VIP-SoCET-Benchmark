# VIP-SoCET-Benchmark

# Description
Purdue University Senior Design with VIP SoCET team. Tasked with creating a ML model for measuring efficiency of hardware optimizations done to a RISC-V processor.

# Results
After running MNIST test data

My implementation accuracy: 95.57%

Tensorflow accuracy: 96.59%

# References
https://github.com/google/gemmlowp

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite

https://arxiv.org/abs/1712.05877


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





