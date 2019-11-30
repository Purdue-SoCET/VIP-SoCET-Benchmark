# VIP-SoCET-Benchmark

# Description
This repository explores the methods used by TensorFlow-Lite for creating machine learning models suitable for integer-only arithmetic. We have written C functions for inferencing using models limited to Conv1D and FullyConnected layers.

# Getting started
Running the make_tflite.py file will create a model.tflite file which can then be used to create the proper header files for running the model using our C code. After running make_tflite.py, run the modelHeaders.py file to update the files in the modelH directory. The .h files in that directory will be used by the C code.

# References
https://github.com/google/gemmlowp

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite

https://arxiv.org/abs/1712.05877
