from keras import Input
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
import glob
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2


def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size,
                 strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size,
                 strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res


def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same',
                  activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and
    #  the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1,
                  padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output],
                  name='generator')
    return model


# Define hyperparameters
data_dir = glob('./DIV2K_train_HR/*')
epochs = 20
batch_size = 1

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Common optimizer for all networks
common_optimizer = Adam(0.0002, 0.5)

generator = build_generator()
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)
generated_high_resolution_images = generator(input_low_resolution)

# print(data_dir)
generator.compile(loss=['binary_crossentropy'], optimizer=common_optimizer)


def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = data_dir

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = cv2.imread(img)
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = cv2.resize(img1, (256, 256))
        img1_low_resolution = cv2.resize(img1, (64, 64))

    return img1_high_resolution, img1_low_resolution


high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                              low_resolution_shape=low_resolution_shape,
                                                              high_resolution_shape=high_resolution_shape)
for epoch in range(epochs):
    print("Epoch:{}".format(epoch))

high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1

low_resolution_images = low_resolution_images.reshape(1, 64, 64, 3)
high_resolution_images = high_resolution_images.reshape(1, 256, 256, 3)

output = generator.fit(low_resolution_images, high_resolution_images, batch_size=batch_size, epochs=epochs)
# output = generator_model.predict(low_resolution_images)
