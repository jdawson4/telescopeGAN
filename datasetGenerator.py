# Author: Jacob Dawson
#
# In this file, we'll be generating the datasets for training. To do this,
# we'll be reading fits files and doing some preprocessing, then `yield`ing
# those data chunks to a keras dataset object. This will be difficult due to
# the fits file format, and the fact that we have several different filters to
# contend with from multiple different sources (Webb, Hubble, etc). For the
# target data, things will be simpler; we'll just be passing some
# already-existing images to the model.
#
# For both raw and prepared data, we might also "cookiecut" them down to size;
# by this I mean that a 800x800 image will get cut into 4 400x400 images
# before being passed to the model. This is so that the model receives the full
# resolution of the data, while also not being overloaded by too large of
# images (there's a real risk of running out of RAM space, for instance, if
# your images are too large)

from astropy.io import fits
import tensorflow as tf
import numpy as np
import os
import gc
from PIL import Image
import imageio.v2 as imageio

# this silences a warning for loading large files (which we do a lot!)
Image.MAX_IMAGE_PIXELS = None

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


def selectNEvenlySpaced(n, arr):
    # this looks ugly but it covers edge cases
    # given an integer n and an array, select n items from the array, evenly
    # spaced as possible
    if n <= 1:
        return arr[0:1]
    if n >= len(arr):
        return arr
    stepsize = len(arr) // (n - 1)
    retArr = []
    for i in range(0, len(arr), stepsize):
        if len(retArr) == n:
            break
        retArr.append(arr[i])
    if len(retArr) < n:
        retArr.append(arr[-1])
    return retArr


def stackImgs(imgs):
    # initialize return object:
    for _, v in imgs.items():
        imgShape = v.shape
        break
    imgShape = [imgShape[0], imgShape[1], numLayers]
    rawData = np.zeros(imgShape)
    blackSingleLayer = rawData[:, :, 0]

    # determine what filters to use:
    filters = []
    for k in imgs.keys():
        filters.append(k)
    filters = sorted(filters)
    filtersToUse = []
    if len(filters) == 0:
        raise Exception("ERROR: No data to stack!")
    elif len(filters) > numLayers:
        j = 0
        for _ in range(0, len(filters), numLayers):
            filtersToUse.append(selectNEvenlySpaced(numLayers, filters[j:]))
            j += 1
    else:
        while len(filters) < numLayers:
            # in case we don't have enough images, pad with black tiles:
            filters.append("blackSingleLayer")
        filtersToUse.append(selectNEvenlySpaced(numLayers, filters))

    # layer and return:
    rawDataList = []
    imgs["blackSingleLayer"] = blackSingleLayer
    for setOfFilters in filtersToUse:
        layer = 0
        rawDataForExport = rawData
        for f in setOfFilters:
            rawDataForExport[:, :, layer] = imgs[f]
            rawDataList.append(rawDataForExport)
            layer += 1
    gc.collect()
    return rawDataList


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
                try:
                    h = hdul[i]
                except:
                    # sometimes might be out of range, just skip this
                    # print("INDEXING ISSUE!")
                    break
                if h.header and not headerFound:
                    header = h.header
                    headerFound = True
                if h.data is None:
                    continue
                if (len(h.data.shape) == 2) and not dataFound:
                    dataFound = True
                    data = h.data
            if not (dataFound and headerFound):
                print("DATA NOT FOUND AT", v)
                continue

            # do our data processing here:
            data = preprocessImg(data)

            telescope = ""
            headerKeys = list(header.keys())
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
            # we actually want the filter to store the wavelength, rather
            # than filter name. Luckily, we can convert easily with an object
            # defined in constants.py:
            if filter in filterDict:
                filter = filterDict[filter]
            else:
                """
                if "INSTRUME" in headerKeys:
                    print(
                        f"FILTER {filter} NOT IN FILTERDICT! INSTRUMENT: {header['INSTRUME']}"
                    )
                """
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
                for r in rawData:
                    # and then there's actually another step, we need to make sure
                    # that the model only receives images of the same size.
                    cookieCutImages = cookieCut(r)
                    for cookieCutImg in cookieCutImages:
                        yield cookieCutImg  # and finally we pass that img to the model!

                # next up: we add our data to the new list:
                imgs = dict()
                imgs[filter] = data
            gc.collect()


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


def determineCardinality(dataset):
    simpleCardinality = dataset.cardinality()
    if simpleCardinality > 0:
        return simpleCardinality
    i = 0
    for _ in dataset.as_numpy_iterator():
        i += 1
    return i


# might seem messy, but we need to do some processing, and set some constants.
# if this file is main, then we're just testing, so print out the cardinality.
# Otherwise, we want to make sure that the file using these functions has the
# cardinalities, so calculate them anyway. In any case, we need to run the
# above dataset generator functions; all that changes is whether we're printing

# first, need to define the signature that all data items returned will be in:
rawReturnSig = tf.TensorSpec(shape=(None, None, numLayers), dtype=tf.uint8)
officialReturnSig = tf.TensorSpec(
    shape=(None, None, numLayersRGB), dtype=tf.uint8
)

# next, make the datasets and determine the cardinalities:
rawDataset = tf.data.Dataset.from_generator(
    lambda: rawDatasetGenerator(),
    output_signature=(rawReturnSig),
)
rawCardinality = determineCardinality(rawDataset)
officialDataset = tf.data.Dataset.from_generator(
    lambda: officialDatasetGenerator(),
    output_signature=(officialReturnSig),
)
officialCardinality = determineCardinality(officialDataset)
datasets = tf.data.Dataset.zip((rawDataset, officialDataset))
# the combined cardinality is just the smaller of the two cardinalities
# combinedCardinality = determineCardinality(datasets) # not necessary
combinedCardinality = (
    rawCardinality
    if rawCardinality < officialCardinality
    else officialCardinality
)

# finally, we apply some behaviors, like asserting the cardinality, batching,
# and prefetching, for smoother training.
# I don't think we're allowed to shuffle, unfortunately
datasets = (
    datasets.apply(tf.data.experimental.assert_cardinality(combinedCardinality))
    .batch(batch_size)
    .prefetch(batch_size * 4)
    .shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
)

if __name__ == "__main__":
    # if this is being run for testing, just print out thoe cardinalities:
    print(f"cardinality of raw: {rawCardinality}")  # printing 993!
    print(f"cardinality of official: {officialCardinality}")  # printing 5272
    print(
        f"cardinality of combined dataset: {combinedCardinality}"
    )  # printing 993
