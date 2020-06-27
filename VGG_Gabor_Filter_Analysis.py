"""
Filename: VGG_Gabor_Filter_Analysis.py
Features: VGG high frequency feature analysis
Author: Mohamed Yasser Imran Zaheer
Last Modified: 24/10/2019
Dependencies: Keras, TensorFlow
Github: https://github.com/yasseriz/SRGAN/tree/yasseriz-final
"""
import keras
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from scipy.misc import imresize, imread
from glob import glob
import matplotlib.pyplot as plt
from scipy.fftpack import fftn
import matplotlib.cm as cm
import cv2


def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = data_dir

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = imread(img, mode='RGB')
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)

        # Do a random horizontal flip
        img1_high_resolution = np.fliplr(img1_high_resolution)
        img1_low_resolution = np.fliplr(img1_low_resolution)

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def build_vgg():
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    input_shape = (256, 256, 3)
    vgg = VGG19(weights="imagenet")
    print(vgg.summary())
    # Set the outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=input_shape)

    # Extract the image features
    img_features = vgg(img)

    return Model(inputs=[img], outputs=[img_features], name='vgg')

# Predict output from VGG network
vgg = build_vgg()
vgg.trainable = False
print(vgg.summary())
vgg.compile(optimizer=Adam(lr=0.0002, beta_1=0.9), loss='mse', metrics=['accuracy'])

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# High and Low resolution inputs to the network
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)

data_dir = glob('./Predict/*')  # glob('./Training_data/*')
batch_size = 1
high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                              high_resolution_shape=high_resolution_shape,
                                                              low_resolution_shape=low_resolution_shape)
# Normalizing images
high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1
features = vgg.predict(high_resolution_images)

# Reshaping images
high_resolution_images = high_resolution_images.reshape(256, 256, 3)
high_resolution_images = 0.5 * high_resolution_images + 0.5
features = 0.5 * features + 0.5

# Building Gabor filter kernel
def build_filters():
    filters = []
    angles = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
        angles.append(theta)
    return filters, angles

# Applying gabor filter kernel as a filter
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(high_resolution_images, -1, kern)
        np.maximum(accum, fimg, accum)
    return accum


filters, angle = build_filters()
angle = np.array(angle)
angle = np.around(angle, decimals=2)

fig = plt.figure(figsize=(15, 15))
c = 1

# Plotting all 256 filter outputs of vgg image
for kern in filters:
    fimg = cv2.filter2D(features[0, :, :, 0], -1, kern)
    ax = fig.add_subplot(4, 4, c)
    ax.imshow(fimg)
    ax.axis("off")
    ax.set_title("Angle: {} rad".format(angle[c - 1]))
    c = c + 1
plt.show()

fig = plt.figure(figsize=(15, 15))
c = 1

# Plotting 16 filtered images from gabor filter original image
for kern in filters:
    fimg = cv2.filter2D(high_resolution_images, -1, kern)
    ax = fig.add_subplot(4, 4, c)
    ax.imshow(fimg)
    ax.axis("off")
    ax.set_title("Angle: {} rad".format(angle[c - 1]))
    c = c + 1
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(high_resolution_images)
ax.axis('off')
ax.set_title('Original Image')
res1 = process(high_resolution_images, filters)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(res1)
ax.axis('off')
ax.set_title('Filtered Image')
plt.show()

# Applying gabor filter to vgg image
def process_vgg(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(features[0, :, :, 0], -1, kern)
        # np.maximum(accum, fimg, accum)
    return fimg

# Plotting VGG images
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(features[0, :, :, 0])
ax.axis('off')
ax.set_title('VGG Image')
res1_vgg = process_vgg(high_resolution_images, filters)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(res1_vgg)
ax.axis('off')
ax.set_title('Filtered Image')
plt.show()

# Plotting HISTOGRAM
plt.figure()
plt.hist(high_resolution_images[:, :, 0])
plt.hist(high_resolution_images[:, :, 1])
plt.hist(high_resolution_images[:, :, 2])
plt.title('Original Image Histogram')
plt.show()

plt.figure()
plt.hist(res1[:, :, 0])
plt.hist(res1[:, :, 1])
plt.hist(res1[:, :, 2])
plt.title('Filtered Image Histogram')
plt.show()

plt.figure()
plt.hist(features[0, :, :, 1], bins=20, density=True, range=[0.2, 1.0])
plt.xlabel('Pixel Intensities')
plt.ylabel('Frequency')
plt.title('VGG Image Histogram')
plt.show()

plt.figure()
plt.hist(res1_vgg)
plt.title('VGG Filtered Image Histogram')
plt.show()
