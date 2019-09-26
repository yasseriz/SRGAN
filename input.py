from comet_ml import Experiment

experiment = Experiment("1Uf990Nvlki77d4AOubsK9lKX", project_name="SRGAN_Personal", log_env_gpu=True)
import time
from keras import Input
from keras.layers import BatchNormalization, Activation, Add, LeakyReLU, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import Adam
import glob
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from scipy.misc import imresize, imread


def write_log(callback, name, value, batch_no):
    """
    Write scalars to Tensorboard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def build_vgg():
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    input_shape = (256, 256, 3)
    vgg = VGG19(weights="imagenet")
    # Set the outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=input_shape)

    # Extract the image features
    img_features = vgg(img)

    return Model(inputs=[img], outputs=[img_features])


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


# Discriminator Network
def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # Last dense layer - for classification
    output_gen = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output_gen], name='discriminator')
    return model


# Adversarial Network
def build_adversarial_model(generator, discriminator, vgg):
    input_low_resolution = Input(shape=(64, 64, 3))

    fake_hr_images = generator(input_low_resolution)
    fake_features = vgg(fake_hr_images)

    discriminator.trainable = False

    output = discriminator(fake_hr_images)

    model = Model(inputs=[input_low_resolution],
                  outputs=[output, fake_features])

    for layer in model.layers:
        print(layer.name, layer.trainable)

    print(model.summary())
    return model


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


# Define hyperparameters
data_dir = glob('./DIV2K_train_HR/*')
epochs = 10000
batch_size = 1

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Common optimizer for all networks
common_optimizer = Adam(0.0002, 0.5)

# Building and compiling the networks
vgg = build_vgg()
vgg.trainable = False
vgg.compile(optimizer=common_optimizer, loss='mse', metrics=['accuracy'])

discriminator = build_discriminator()
discriminator.compile(optimizer=common_optimizer, loss='mse', metrics=['accuracy'])

generator = build_generator()
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)
generated_high_resolution_images = generator(input_low_resolution)

features = vgg(generated_high_resolution_images)
discriminator.trainable = False
discriminator.compile(optimizer=common_optimizer, loss='mse', metrics=['accuracy'])

probs = discriminator(generated_high_resolution_images)

adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
adversarial_model.compile(optimizer=common_optimizer, loss=['binary_crossentropy', 'mse'], metrics=['accuracy'],
                          loss_weights=[1e-3, 1])
# Add Tensorboard
tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)

# for epoch in range(epochs):
# print("Epoch:{}".format(epoch))

# inpu = cv2.cvtColor(low_resolution_images, cv2.COLOR_BGR2RGB)
# plt.imshow(inpu)
# plt.show()
# cv2.imshow('inpu', inpu)
# cv2.waitKey(0)


# print(len(low_resolution_images))

# Training
# for epoch in range(epochs):
#     print("Epoch :{}".format(epoch))
#     experiment.log_parameter('epoch', epoch)
#
#     # Training the discriminator network
#
#     high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
#                                                                   low_resolution_shape=low_resolution_shape,
#                                                                   high_resolution_shape=high_resolution_shape)
#
#     high_resolution_images = high_resolution_images / 127.5 - 1
#     low_resolution_images = low_resolution_images / 127.5 - 1
#
#     low_resolution_images = low_resolution_images.reshape(1, 64, 64, 3)
#     high_resolution_images = high_resolution_images.reshape(1, 256, 256, 3)
#
#     generated_high_resolution_images = generator.predict(low_resolution_images)
#
#     # Generating batch of real and fake labels
#     real_labels = np.ones((batch_size, 16, 16, 1))
#     fake_labels = np.zeros((batch_size, 16, 16, 1))
#
#     # d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
#     # d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
#     #
#     # # Calculating the discriminator loss
#     # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#     # print("d_loss :", d_loss)
#
#     # Training the generator network
#     high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
#                                                                   low_resolution_shape=low_resolution_shape,
#                                                                   high_resolution_shape=high_resolution_shape)
#
#     high_resolution_images = high_resolution_images / 127.5 - 1
#     low_resolution_images = low_resolution_images / 127.5 - 1
#
#     low_resolution_images = low_resolution_images.reshape(1, 64, 64, 3)
#     high_resolution_images = high_resolution_images.reshape(1, 256, 256, 3)
#
#     image_features = vgg.predict(high_resolution_images)
#
#     # g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
#     #                                           [real_labels, image_features])
#     # print("g_loss :", g_loss)
#     # # Write the losses to Tensorboard
#     # write_log(tensorboard, 'g_loss', g_loss[0], epoch)
#     # write_log(tensorboard, 'd_loss', d_loss[0], epoch)
#     #
#     # generator.save_weights("generator.h5")
#     # discriminator.save_weights("discriminator.h5")


discriminator.load_weights("discriminator.h5")
generator.load_weights("generator.h5")

high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                              low_resolution_shape=low_resolution_shape,
                                                              high_resolution_shape=high_resolution_shape)

high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1

low_resolution_images = low_resolution_images.reshape(1, 64, 64, 3)
high_resolution_images = high_resolution_images.reshape(1, 256, 256, 3)

generated_images = generator.predict_on_batch(low_resolution_images)


low_resolution_images = low_resolution_images.reshape(64, 64, 3)
high_resolution_images = high_resolution_images.reshape(256, 256, 3)
generated_images = generated_images.reshape(256, 256, 3)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.imshow(low_resolution_images)
ax.axis("off")
ax.set_title("Low-resolution")

ax = fig.add_subplot(1, 3, 2)
ax.imshow(high_resolution_images)
ax.axis("off")
ax.set_title("Original")

ax = fig.add_subplot(1, 3, 3)
ax.imshow(generated_images)
ax.axis("off")
ax.set_title("Generated")

plt.show()












# generator.fit(low_resolution_images, high_resolution_images, batch_size=batch_size, epochs=epochs)

# generator.save_weights('model.h5')
# generator.load_weights('model.h5')
# output = generator.predict(low_resolution_images)
# print(output)
# output = output.reshape((256, 256, 3))
# out = 0.5 * output + 0.5
# img_hr = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
# plt.imshow(img_hr)
# plt.show()


# cv2.imshow('img_hr', img_hr)
# cv2.waitKey(0)

experiment.end()
