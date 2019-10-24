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
        if np.random.random() < 0.5:
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

data_dir = glob('./Training_data/*')
batch_size = 1
high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                              high_resolution_shape=high_resolution_shape,
                                                              low_resolution_shape=low_resolution_shape)
high_resolution_images = high_resolution_images / 127.5 - 1
low_resolution_images = low_resolution_images / 127.5 - 1
# print(high_resolution_images.shape)
features = vgg.predict(high_resolution_images)

### NEW METHOD ###
high_resolution_images = high_resolution_images.reshape(256, 256, 3)
high_resolution_images = 0.5 * high_resolution_images + 0.5

def build_filters():
    filters = []
    angle = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((31, 31), 10, theta, 30, 0.25, 0, ktype=cv2.CV_32F)
        # kern /= 1.5 * kern.sum()
        filters.append(kern)
        angle.append(theta)
    return filters, angle


filters, angle = build_filters()
angle = np.array(angle)
angle = np.around(angle, decimals=2)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(high_resolution_images)
ax.axis("off")
ax.set_title('Original Image')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(features[0, :, :, 1])
ax.axis("off")
ax.set_title('VGG Output')
plt.show()

fig = plt.figure(figsize=(15, 15))
c = 1

for kern in filters:
    fimg = cv2.filter2D(features[0, :, :, 1], -1, kern)
    ax = fig.add_subplot(4, 4, c)
    ax.imshow(fimg)
    ax.axis("off")
    ax.set_title("Angle: {} rad".format(angle[c-1]))
    c = c + 1
#
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(high_resolution_images)
# ax.axis("off")
# ax.set_title('Original Image')
#
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(fimg)
# ax.axis("off")
# ax.set_title('HR after Filter')
plt.show()


# print(dst.shape)
# dst = np.expand_dims(dst, axis=0)
# d_features = vgg.predict(dst)
# dst = np.squeeze(dst, axis=0)
# print(d_features.shape)
#
# t = cv2.filter2D(d_features[0, :, :, 1], -1, test)
# print(t.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(2,2,1)
# ax.imshow(high_resolution_images)
# ax.axis("off")
# ax.set_title('Original Image')
#
# ax = fig.add_subplot(2,2,2)
# ax.imshow(dst)
# ax.axis("off")
# ax.set_title('HR after Filter')
#
# ax = fig.add_subplot(2,2,3)
# ax.imshow(d_features[0, :, :, 1])
# ax.axis("off")
# ax.set_title('After VGG')
#
# ax = fig.add_subplot(2,2,4)
# ax.imshow(t)
# ax.axis("off")
# ax.set_title('Filter after VGG')
# plt.show()

def build_filters():
    filters = []
    angle = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((21, 21), 10, theta, 30, 0.25, 0, ktype=cv2.CV_32F)
        # kern /= 1.5 * kern.sum()
        filters.append(kern)
        angle.append(theta)
    return filters, angle


filters, angle = build_filters()
angle = np.array(angle)
angle = np.around(angle, decimals=2)
fig = plt.figure(figsize=(15, 15))
c = 1

for kern in filters:
    fimg = cv2.filter2D(features[0, :, :, 1], -1, kern)
    ax = fig.add_subplot(4, 4, c)
    ax.imshow(fimg)
    ax.axis("off")
    ax.set_title("Angle: {} rad".format(angle[c-1]))
    c = c + 1

plt.show()
