# Author: Jacob Dawson
#
# This file will take all of our raw fits data, lightly preprocess it, and save
# the data to simple images for processing by our generator. The hope is that
# these images are small enough for our generator to train on full-res images,
# but without all the crap (black space).
#
# We will also be cookiecutting our official images down to size, but this is
# a much simpler process because we can access the raw pngs rather than have
# to fiddle with weird over/underexposed fits files.

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import imageio
from constants import *

onlyfiles = [f for f in listdir('raw_data') if isfile(join('raw_data', f))]
imgCount=0
for file in onlyfiles:
    with fits.open('raw_data/' + file) as hdul:
        data = hdul[1].data
        data = np.clip(data, np.percentile(data, 20.0), np.percentile(data, 99.625))
        if (np.amin(data) < 0.0):
            data += np.amin(data)
        if (np.amin(data) > 0.0):
            data -= np.amin(data)
        data /= np.amax(data)
        data *= 255.0
        for i in range(0, data.shape[0]-image_size, image_size):
            for j in range(0, data.shape[1]-image_size, image_size):
                cut_img = data[i:i+image_size, j:j+image_size]
                if np.mean(cut_img) > 10:
                    imgCount+=1
                    imageio.imwrite('cookiecut_raw_data/'+str(imgCount)+'_'+str(i)+'_'+str(j)+'.png', cut_img)

"""
for imgName in listdir('official_images/'):
    img = imageio.imread('official_images/'+imgName, pilmode='RGB')
    for i in range(0, img.shape[0]-image_size, image_size):
        for j in range(0, img.shape[1]-image_size, image_size):
            cut_img = img[i:i+image_size, j:j+image_size]
            if np.mean(cut_img) > 10:
                imageio.imwrite('cookiecut_official_images/'+imgName+'_'+str(i)+'_'+str(j)+'.png', cut_img)
"""
