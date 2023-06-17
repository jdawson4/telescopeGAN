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
import gc
import random

numLayers = 4


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
    data = data.astype(np.uint8)
    gc.collect()
    return data


def stackImgs(imgs):
    imgShape = imgs.values()[0].shape
    imgShape = [imgShape[0], imgShape[1], numLayers]
    rawData = np.zeros(imgShape)
    filters = []
    for k in imgs.keys():
        filters.append(k)
    # let's just chose n random filters from that list:
    filtersToUse = random.choices(filters, k=numLayers)
    layer = 0
    for f in filtersToUse:
        rawData[:, :, layer] = imgs[f]
    gc.collect()
    return rawData


def datasetGenerator(folder):
    list_of_files = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(".fits"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    prevTarget = "none"
    imgs = dict()

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
                    continue
            if not dataFound:
                print("DATA NOT FOUND AT", v)
                continue

            # do our data processing here:
            data = preprocessImg(data)

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
                continue

            # let's also determine what we're looking at. Apparently targname
            # is standard across all telescopes (though I've only looked at
            # JWST and HST)
            if "TARGNAME" not in headerKeys:
                print("COULD NOT FIND TARGET NAME!")
                continue
            # assume we've found it from this point on:
            target = header["TARGNAME"]

            # handle base case:
            if prevTarget == "none":
                prevTarget = target

            if target == prevTarget:
                imgs[filter] = data
                gc.collect()
            else:
                # we're going to process the previous target's data, then pass
                # our new image into the dict.
                rawData = stackImgs(imgs)
                yield rawData  # and finally we pass that img to the model!

                # next up: we add our data to the new list:
                imgs = dict()
                imgs[filter] = data

    rawData = stackImgs(imgs)
    yield rawData  # and finally we pass that img to the model!
