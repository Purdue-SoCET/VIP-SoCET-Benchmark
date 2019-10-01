#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
import uuid
import model_pb2


"""
adated from:
https://github.com/transcranial/keras-js/blob/master/python/encoder.py
"""


def quantize_arr(arr, dtype=np.uint8):
    """Quantization based on real_value = scale * (quantized_value - zero_point).
    """
    min_val, max_val = np.min(arr), np.max(arr)

    scale, zero_point = choose_quant_params(min_val, max_val, dtype=dtype)

    transformed_arr = zero_point + arr / scale
    # print(transformed_arr)
    if dtype == np.uint8:
        clamped_arr = np.clip(transformed_arr, 0, 255)
        quantized = clamped_arr.astype(np.uint8)
    elif dtype == np.uint32:
        clamped_arr = np.clip(transformed_arr, -2147483647, 2147483647)
        quantized = clamped_arr.astype(np.uint32)
    else:
        raise ValueError('dtype={} is not supported'.format(dtype))

    # print(clamped_arr)
    min_val = min_val.astype(np.float32)
    max_val = max_val.astype(np.float32)

    return quantized, min_val, max_val


def choose_quant_params(min, max, dtype=np.uint8):
    # Function adapted for python from:
    # https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
    # We extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    min = np.min([min, 0])
    max = np.max([max, 0])

    # the min and max quantized values, as floating-point values
    if dtype == np.uint8:
        qmin = 0.0
        qmax = 255.0
    elif dtype == np.int32:
        qmin = -2147483647.0
        qmax = 2147483647.0
    else:
        raise ValueError('dtype={} is not supported'.format(dtype))

    # First determine the scale.
    scale = (max - min) / (qmax - qmin)

    # Zero-point computation.
    # First the initial floating-point computation. The zero-point can be
    # determined from solving an affine equation for any known pair
    # (real value, corresponding quantized value).
    # We know two such pairs: (rmin, qmin) and (rmax, qmax).
    # Let's use the first one here.
    initial_zero_point = qmin - min / scale

    # Now we need to nudge the zero point to be an integer
    # (our zero points are integer, and this is motivated by the requirement
    # to be able to represent the real value "0" exactly as a quantized value,
    # which is required in multiple places, for example in Im2col with SAME
    # padding).

    if initial_zero_point < qmin:
        nudged_zero_point = np.uint8(qmin)
    elif initial_zero_point > qmax:
        nudged_zero_point = np.uint8(qmax)
    else:
        nudged_zero_point = np.uint8(round(initial_zero_point))

    return scale, nudged_zero_point


class KerasQuantizer:
    """Encoder class.
    Takes as input a Keras model saved in hdf5 format that includes the model architecture with the weights.
    This is the resulting file from running the command:
    ```
    model.save('my_model.h5')
    ```
    See https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
    """

    def __init__(self, hdf5_model_filepath):
        if not hdf5_model_filepath:
            raise Exception('hdf5_model_filepath must be provided.')
        self.hdf5_model_filepath = hdf5_model_filepath
        self.layers = {}

        self.create_model()
        self.quantize()

    def create_model(self):
        """Initializes a model from the protobuf definition.
        """
        self.model = model_pb2.Model()
        self.model.id = str(uuid.uuid4())
        self.model.name = 'MyModel'

    def quantize(self):
        """quantize method.
        Originally was serialize function from encoder.py
        Strategy for extracting the weights is adapted from the
        load_weights_from_hdf5_group method of the Container class:
        see https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L2505-L2585
        """

        hdf5_file = h5py.File(self.hdf5_model_filepath, mode='r')

        self.model.keras_version = hdf5_file.attrs['keras_version']
        self.model.backend = hdf5_file.attrs['backend']
        self.model.model_config = hdf5_file.attrs['model_config']

        f = hdf5_file['model_weights']
        for layer_name in f.attrs['layer_names']:
            g = f[layer_name]
            self.layers[layer_name] = {}
            for weight_name in g.attrs['weight_names']:
                weight_value = g[weight_name].value
                if b'/bias:0' in weight_name:
                    quant_type = np.int32
                else:
                    quant_type = np.uint8
                quantized, min_val, max_val = quantize_arr(weight_value, dtype=quant_type)
                self.layers[layer_name][weight_name] = quantized.astype(quant_type)
        hdf5_file.close()
