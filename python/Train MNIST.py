from keras.datasets.mnist import load_data
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from pprint import pprint as pp
import keras.backend as K
import tensorflow
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# the train set & the other for the test set
(train_digits, train_labels), (test_digits, test_labels) = load_data()
image_height = train_digits.shape[1]
image_width = train_digits.shape[2]
num_channels = 1  # we have grayscale images


def build_model(scale, num_classes):

    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=scale, kernel_size=(3,3), activation='sigmoid', padding='same', input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=scale*2, kernel_size=(3,3), activation='sigmoid', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=scale*2, kernel_size=(3,3), activation='sigmoid', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(scale*4, activation='sigmoid'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # re-shape the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))
    test_data = np.reshape(test_digits, (test_digits.shape[0],image_height, image_width, num_channels))

    # re-scale the image data to values between (0.0,1.0]
    train_data = train_data.astype('float32') / 255.
    test_data = test_data.astype('float32') / 255.

    # one-hot encode the labels - we have 10 output classes
    # so 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0] & so on
    from keras.utils import to_categorical
    num_classes = 10
    train_labels_cat = to_categorical(train_labels,num_classes)
    test_labels_cat = to_categorical(test_labels,num_classes)
    #train_labels_cat.shape, test_labels_cat.shape


    # shuffle the training dataset (5 times!)
    for _ in range(5):
        indexes = np.random.permutation(len(train_data))

    train_data = train_data[indexes]
    train_labels_cat = train_labels_cat[indexes]

    # now set-aside 10% of the train_data/labels as the
    # cross-validation sets
    val_perc = 0.10
    val_count = int(val_perc * len(train_data))

    # first pick validation set from train_data/labels
    val_data = train_data[:val_count,:]
    val_labels_cat = train_labels_cat[:val_count,:]

    # leave rest in training set
    train_data2 = train_data[val_count:,:]
    train_labels_cat2 = train_labels_cat[val_count:,:]

    # NOTE: We will train on train_data2/train_labels_cat2 and
    # cross-validate on val_data/val_labels_cat
    model = build_model(16, num_classes)
    print(model.summary())

    results = model.fit(train_data2, train_labels_cat2,
                        epochs=3, batch_size=64,
                        validation_data=(val_data, val_labels_cat))

    test_loss, test_accuracy = \
      model.evaluate(test_data, test_labels_cat, batch_size=64)
    print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
    model.save('sigmoid_trained.h5')
    return model

if __name__ == '__main__':
    model = train_model()
    #model = load_model('partly_trained.h5')
    i = 0
    layers = []
    for layer in model.layers:
        w8s = []
        biases = []
        weights = layer.get_weights()  # list of numpy arrays
        if weights is not None:
            if len(weights) >= 1:
                w8s = ((weights[0] + 1) * 255) / 2
                w8s = w8s.round()
                w8s = w8s.astype('uint8')
            if len(weights) >= 2:
                biases = ((weights[1] + 1) * 255) / 2
                biases = biases.round()
                biases = biases.astype('uint8')
            new_layer = [w8s, biases]
            layers.append(new_layer)

    new_layers = []
    for layer in layers:
        if len(layer[0]) > 0:
            weights = layer[0]
            biases = layer[1]
            weights = weights.flatten()
            biases = biases.flatten()
            new_layers = np.hstack([new_layers, weights, biases])


    num_bins = 10
    n, bins, patches = plt.hist(new_layers, num_bins, facecolor='blue', alpha=0.5, edgecolor='black', linewidth=1)
    plt.show()




