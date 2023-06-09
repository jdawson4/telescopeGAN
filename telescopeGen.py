# Author: Jacob Dawson
# just a simple script for generating images using the trained GAN.

import tensorflow as tf
import numpy as np
import imageio
from astropy.io import fits
from os import listdir
from os.path import isfile, join
from constants import *


def determine_padding(image):
    downscalinglayers = 16
    i = image.shape[0]
    j = image.shape[1]
    while (i % downscalinglayers) != 0:
        i += 1
    while (j % downscalinglayers) != 0:
        j += 1
    return i, j


trained_gen = tf.keras.models.load_model("telescopeGen")
onlyfiles = [f for f in listdir("raw_data") if isfile(join("raw_data", f))]
imgCount = 0
for file in onlyfiles:
    with fits.open("raw_data/" + file) as hdul:
        data = hdul[1].data
        data = np.clip(data, np.percentile(data, 20.0), np.percentile(data, 99.625))
        if np.amin(data) < 0.0:
            data += np.amin(data)
        if np.amin(data) > 0.0:
            data -= np.amin(data)
        data /= np.amax(data)
        data *= 255.0
        raw_img = tf.image.grayscale_to_rgb(
            tf.constant(data, shape=(len(data), len(data[1]), 1))
        )
        x, y = determine_padding(raw_img)
        raw_img = tf.image.resize_with_crop_or_pad(raw_img, x, y)
        raw_imgs = tf.expand_dims(raw_img, axis=0)
        fake_images = trained_gen(raw_imgs)
        fake_images = tf.cast(fake_images, tf.float16)
        fake_images = fake_images.numpy().astype(np.uint8)
        imageio.imwrite("fake_images/" + str(file) + ".png", fake_images[0])
