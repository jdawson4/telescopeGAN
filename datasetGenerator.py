# Author: Jacob Dawson
#
# In this file, we'll be generating the datasets for training. To do this,
# we'll be reading fits files and doing some preprocessing, then `yield`ing
# those data chunks to a keras dataset object. This will be difficult due to
# the fits file format, and the fact that we have several different filters to
# contend with from multiple different sources (Webb, Hubble, etc). For the
# target data, things will be simpler; we'll just be passing some
# already-existing images to the model.

from astropy.io import fits

# import tensorflow as tf
import numpy as np
import os


def preprocessImg(data):
    data = data.astype(np.float32)
    data = np.nan_to_num(data)
    data = np.clip(data, np.percentile(data, 20.0), np.percentile(data, 99.625))
    if np.amin(data) < 0.0:
        data += np.amin(data)
    if np.amin(data) > 0.0:
        data -= np.amin(data)
    data /= np.amax(data)
    data *= 255.0
    return data


def datasetGenerator(folder):
    list_of_files = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(".fits"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    for _, v in list_of_files.items():
        with fits.open(v) as hdul:
            dataFound = False
            headerFound = False
            for i in range(3):
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
                print("DATA NOT FOUND AT", v)
                break

            telescope = ""
            headerKeys = header.keys()
            if "FILTER" in headerKeys:
                # seems like this is a JWST img
                telescope = "JWST"
                filter = header["FILTER"]
            elif "FILTNAM1" in headerKeys:
                # seems like this is a Hubble img
                telescope = "HUBBLE"
                filter = header["FILTNAM1"]
            else:
                # In this case, I was wrong--not sure what the filter is
                print("COULD NOT FIND FILTER!")
                break
