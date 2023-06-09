# Author: Jacob Dawson
#
# I'm just using this file to test/demonstrate some simply functionality with
# astropy on our data. The whole point of this repo is to kinda get used to
# using astropy, especially in an ml environment, so this is an important
# testing-ground

from astropy.io import fits
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def processFile(path):
    with fits.open(path) as hdul:
        dataFound = False
        for i in range(2):
            h = hdul[i]
            if h.data is None:
                continue
            if len(h.data.shape) == 2:
                dataFound = True
                data = h.data
                header = h.header
                break
        if not dataFound:
            print("DATA NOT FOUND AT", path)
            return

        # let's get the filter from the header:
        print(header["FILTNAM1"])  # something like 'F284W', normally
        # print(header['FILTNAM2'])
        # print(header['FILTER1'])
        # print(header['FILTER2'])
        # print(header['PFILTER1'])
        # print(header['PFILTER2'])

        # get some basic facts about our data:
        # print(np.amax(data)) # 4154.5635
        # print(np.amin(data)) # -1340.3322
        # print(np.mean(data)) # 658.7233

        # and do our processing:
        data = data.astype(np.float32)
        data = np.nan_to_num(data)
        data = np.clip(data, np.percentile(data, 20.0), np.percentile(data, 99.625))
        if np.amin(data) < 0.0:
            data += np.amin(data)
        if np.amin(data) > 0.0:
            data -= np.amin(data)
        data /= np.amax(data)
        data *= 255.0

        testData = tf.constant(data, shape=(len(data), len(data[1]), 1))
        print(testData.shape)
        # imgplot = plt.imshow(data, cmap="gray")
        # plt.show()


list_of_files = {}
for dirpath, dirnames, filenames in os.walk("example_data"):
    for filename in filenames:
        if filename.endswith(".fits"):
            list_of_files[filename] = os.sep.join([dirpath, filename])

for _, v in list_of_files.items():
    processFile(v)
    break
