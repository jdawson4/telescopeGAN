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
        headerFound = False
        for i in range(2):
            h = hdul[i]
            if h.header and not headerFound:
                header = h.header
                headerFound = True
            if h.data is None:
                continue
            if len(h.data.shape) == 2:
                dataFound = True
                data = h.data
                break
        if not dataFound:
            print("DATA NOT FOUND AT", path)
            return

        # for k, v in header.items():
        #    print(k)
        # important finding: Webb images have the field "FILTER" in their
        # header; Hubble has the field "FILTNAM1"

        # let's get the filter from the header:
        headerKeys = header.keys()
        if "FILTER" in headerKeys:
            # seems like this is a JWST img
            filter = header["FILTER"]
            # print(header["FILTER"])
        elif "FILTNAM1" in headerKeys:
            # seems like this is a Hubble img
            filter = header["FILTNAM1"]
            # print(header["FILTNAM1"])
        else:
            # In this case, I was wrong--not sure what the filter is
            print("COULD NOT FIND FILTER!")
            return

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

        return filter


list_of_files = {}
for dirpath, dirnames, filenames in os.walk("raw_data"):
    for filename in filenames:
        if filename.endswith(".fits"):
            list_of_files[filename] = os.sep.join([dirpath, filename])

filters = []
for _, v in list_of_files.items():
    filt = processFile(v)
    if filt != None and filt not in filters:
        filters.append(filt)
    # break
print(filters)
