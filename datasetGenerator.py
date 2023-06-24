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

import tensorflow as tf
import numpy as np
import os
import gc
import random
import imageio

from constants import *


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


def cookieCut(fullImg):
    # To cut down on processing, we'll "cookie cut" those down to a
    # reasonable size; this also removes the black space that many
    # images have. This ensures that our model has a consistently
    # filled, yet limited size of data for processing!
    cookieCutImages = []
    for i in range(0, fullImg.shape[0] - image_size, image_size):
        for j in range(0, fullImg.shape[1] - image_size, image_size):
            cut_img = fullImg[i : i + image_size, j : j + image_size, :]
            if np.mean(cut_img) > 5:
                cookieCutImages.append(cut_img)
    return cookieCutImages


def rawDatasetGenerator():
    list_of_files = {}
    for dirpath, dirnames, filenames in os.walk(rawImgDir):
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
            else:
                # we're going to process the previous target's data, then pass
                # our new image into the dict.
                rawData = stackImgs(imgs)

                # and then there's actually another step, we need to make sure
                # that the model only receives images of the same size.
                cookieCutImages = cookieCut(rawData)
                for cookieCutImg in cookieCutImages:
                    yield cookieCutImg  # and finally we pass that img to the model!

                # next up: we add our data to the new list:
                imgs = dict()
                imgs[filter] = data
            gc.collect()

    # we actually still have "one in the chamber" so to speak, so let's process
    # that one and we're done!
    rawData = stackImgs(imgs)
    cookieCutImages = cookieCut(rawData)
    for cookieCutImg in cookieCutImages:
        yield cookieCutImg


# now that all of that nasty business is over, we also want to make a dataset
# generator for the "official" images, aka our "target" or "true data
# distribution" that the generator will be aiming to mimic, and that the
# discriminator will be hoping to positively identify.
def officialDatasetGenerator():
    for imgName in os.listdir(officialImgDir):
        img = imageio.imread(officialImgDir + imgName, pilmode="RGB")
        img = img.astype(np.uint8)  # should already be uint8, but make sure
        cookieCutImgs = cookieCut(img)
        for cookieCutImg in cookieCutImgs:
            yield cookieCutImg
        gc.collect()


# ok I'm a bit confused about how datasets work in keras. Some experimenting:
if __name__ == "__main__":
    # need to define the signature that all data items returned will be in:
    rawReturnSig = tf.TensorSpec(shape=(None, None, numLayers), dtype=tf.uint8)
    officialReturnSig = tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)

    # first let's figure out how the cardinality of this one works:
    rawDataset = tf.data.Dataset.from_generator(
        lambda: rawDatasetGenerator(),
        output_signature=(rawReturnSig),
    )
    print(f"cardinality of raw: {rawDataset.cardinality()}")

    # next work out the cardinality of the second dataset
    officialDataset = tf.data.Dataset.from_generator(
        lambda: officialDatasetGenerator(),
        output_signature=(officialReturnSig),
    )
    print(f"cardinality of official: {officialDataset.cardinality()}")

    # we also need to zip these together for the model:
    # datasets = tf.data.Dataset.zip((rawDataset, officialDataset))
