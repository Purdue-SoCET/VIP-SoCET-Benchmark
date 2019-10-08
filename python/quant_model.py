from encoder import Encoder
import numpy as np


IMG_SHAPE = (28, 28)


class QuantModel:
    """
    Integer representation of keras model
    """

    def __init__(self, hdf5_model_filepath, activation_params):
        if not hdf5_model_filepath:
            raise Exception('hdf5_model_filepath must be provided.')
        if not activation_params:
            raise Exception('activation_params must be provided.')

        self.hdf5_model_filepath = hdf5_model_filepath
        self.layers = []
        self.act_params = activation_params
        self.load_quant_model()

    def load_quant_model(self):
        quant_model = Encoder(self.hdf5_model_filepath, 'quant_model', 1)
        quant_model.serialize()

        parameters_conv2d = quant_model.layers[b'conv2d']
        weights_conv2d = parameters_conv2d[b'conv2d/kernel:0']
        biases_conv2d = parameters_conv2d[b'conv2d/bias:0']

        parameters_dense_1 = quant_model.layers[b'dense']
        weights_dense_1 = parameters_dense_1[b'dense/kernel:0']
        biases_dense_1 = parameters_dense_1[b'dense/bias:0']

        parameters_dense_2 = quant_model.layers[b'dense_1']
        weights_dense_2 = parameters_dense_2[b'dense_1/kernel:0']
        biases_dense_2 = parameters_dense_2[b'dense_1/bias:0']

        self.layers = [[weights_conv2d, biases_conv2d], [weights_dense_1, biases_dense_1], [weights_dense_2, biases_dense_2]]

    def infer(self):
        pass


def _activation_func(x):
    return np.max([0, x])


def _evaluate_conv_layer(input_layer, weights, biases):
    input_layer = np.reshape(input_layer, IMG_SHAPE)
    filters = weights.shape[3]
    output_shape = (filters, IMG_SHAPE[0], IMG_SHAPE[1])
    output = np.zeros(shape=output_shape)
    for i in range(filters):
        for j in range(0, IMG_SHAPE[0] - 2):
            for k in range(0, IMG_SHAPE[1] - 2):
                kernel_img = input_layer[j:j + 3, k:k + 3]
                kernel_weight = weights[:, :, 0, i]
                result = np.dot(kernel_img, kernel_weight) + biases[i]
                output[i][j + 1][k + 1] = _activation_func(result)
    return output


def _evaluate_dense_layer(input, weights, biases, num_channels):
    output = np.zeros(shape=(num_channels,))
    for i in range(num_channels):
        output[i] = _activation_func(np.sum(input * weights[:, i]) + biases[i])
    return output


def evaluate_pred_layer(input, weights, biases, num_channels):
    output = np.zeros(shape=(num_channels,))
    for i in range(num_channels):
        output[i] = np.sum(input * weights[:, i]) + biases[i]
    # Softmax prediction
    output_exp = np.exp(output)
    total = np.sum(output_exp)
    for i in range(num_channels):
        output[i] = output_exp[i] / total
    return output

