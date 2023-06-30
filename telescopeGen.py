# Author: Jacob Dawson
# just a simple script for generating images using the trained GAN.

import tensorflow as tf
import numpy as np
import imageio
from astropy.io import fits
from os import listdir
from os.path import isfile, join
from constants import *
from datasetGenerator import *


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
imgCount = 0
for rawImg in rawDataset.as_numpy_iterator():
    imgCount+=1
    raw_image = tf.convert_to_tensor(rawImg, dtype=tf.float32)
    fake_image = trained_gen(
        tf.expand_dims(raw_image, 0), training=False
    )[0]
    fake_image = fake_image.numpy().astype(np.uint8)
    imageio.imwrite(
        fakeImageDir + str(imgCount) + ".png", fake_image
    )
