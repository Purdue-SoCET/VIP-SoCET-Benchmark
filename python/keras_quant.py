import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from encoder import choose_quant_params, Encoder
import random


def get_hidden_layer(model, layer, data):
    """
    Loads MNIST data
    """
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
    intermediate_output = intermediate_layer_model.predict(data)
    return intermediate_output


def get_scale_zero_points(arr):
    min, max = np.min(arr), np.max(arr)
    scale, zero_points = choose_quant_params(min, max)
    return scale, zero_points


def load_mnist():
    """
    Loads MNIST data
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = to_categorical(y_train, num_classes=10)  # one hot
    y_test = to_categorical(y_test, num_classes=10)  # one hot

    return x_train, x_test, y_train, y_test


def train_load_model(filename, train=True):
    if not train:
        return load_model(filename)
    # Create model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, name='pred', activation='softmax'))

    # Quantization aware training
    sess = tf.keras.backend.get_session()
    tf.contrib.quantize.create_training_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    # You can plot the quantize training graph on tensorboard
    # tf.summary.FileWriter('/workspace/tensorboard', graph=sess.graph)

    # Define optimizer
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # We add metrics to get more results you want to see
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=256)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    model.save(filename)

    # quant_param[layers][weight/activation][min/max]
    quant_params = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    i = 0

    # Print the min max values from fakequant
    for node in sess.graph.as_graph_def().node:
        if 'weights_quant/AssignMaxLast' in node.name:
            tensor = sess.graph.get_tensor_by_name(node.name + ':0')
            print('{} = {}'.format(node.name, sess.run(tensor)))
            quant_params[i][0][1] = sess.run(tensor)
        if 'weights_quant/AssignMinLast' in node.name:
            tensor = sess.graph.get_tensor_by_name(node.name + ':0')
            print('{} = {}'.format(node.name, sess.run(tensor)))
            quant_params[i][0][0] = sess.run(tensor)
        if 'act_quant/min/Assign' in node.name:
            tensor = sess.graph.get_tensor_by_name(node.name + ':0')
            print('{} = {}'.format(node.name, sess.run(tensor)))
            quant_params[i][1][1] = sess.run(tensor)
        if 'act_quant/max/Assign' in node.name:
            tensor = sess.graph.get_tensor_by_name(node.name + ':0')
            print('{} = {}'.format(node.name, sess.run(tensor)))
            quant_params[i][1][1] = sess.run(tensor)
            i += 1
    return model, quant_params


x_train, x_test, y_train, y_test = load_mnist()

model, quant_params = train_load_model('lastest.h5', train=True)

for layer in quant_params:
    for weight_act in layer:
        weight_act[0], weight_act[1] = choose_quant_params(weight_act[0], weight_act[1])


