# Author: Jacob Dawson
# this file contains a bunch of constants that control the construction
# and processing of our data. These include hyperparameters and certain
# numbers that control preprocessing.
#
# CONSTANTS
seed = 3  # my lucky number!
batch_size = 16  # higher obviously better. I'd recommend making the number of items in the trainset divisible by this. Otherwise... issues
image_size = 400
numLayers = 4  # the number of greyscale inputs to include
numLayersRGB = 3  # three layers, for RGB images
epochs = 100
rawImgDir = "raw_data/"
officialImgDir = "official_images/"
fakeImageDir = "fake_images/"
checkPointImageDir = "checkpoint_images/"

# custom hyperparameters--determine things about loss:
chi = 0.75  # how much we care about SSIM vs L1 when creating content loss
# ^ not sure if we're even going to use L2 after all.
content_lambda = 0.375  # content loss weight
wgan_lambda = 1.0  # the weight we give to fooling the wgan

# learning rates: a few different strategies:
# 1.
# idea here comes from https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628, called TTUR
gen_learn_rate = 0.001
dis_learn_rate = 0.002
# 2.
# this idea I've seen more frequently, the idea is to give the generator more ability to shift rapidly to outwit the discriminator
# gen_learn_rate = 0.0001
# dis_learn_rate = 0.00001
# 3.
# Very low for both
# gen_learn_rate = 0.001
# dis_learn_rate = 0.001

# no idea what effect this has!
momentum = 0.9  # default for adam is 0.9, default for RMSProp is 0.0

# here's where we're going to have to store all kinds of data about filters.
# I wanted to avoid this ugliness but I don't think I can. Ugh.
# Please note that this maps a filter name to pivot wavelength in Angstroms
filterDict = {
    # long pass and extremely wide Hubble:
    "F200LP": 4971.9,
    "F300X": 2820.5,
    "F350LP": 5873.9,
    "F475X": 4940.7,
    "F600LP": 7468.1,
    "F850LP": 9176.1,
    # wide-band Hubble filters:
    "F218W":2228,
    "F225W":2372.1,
    "F275W":2709.7,
    "F336W":3354.5,
    "F390W":3923.7,
    "F438W":4326.2,
    "F475W":4773.1,
    "F555W":5308.4,
    "F606W":5889.2,
    "F625W":6242.6,
    "F775W":7651.4,
    "F814W":8039.1,
    # medium-band Hubble filters:
    "F390M":3897.2,
    "F410M":4109,
    "FQ422M":4219.2,
    "F467M":4682.6,
    "F547M":5447.5,
    "F621M":6218.9,
    "F689M":6876.8,
    "F763M":7614.4,
    "F845M":8439.1,
    # narrow-band Hubble filters:
    "FQ232N":2432.2,
    "FQ243N":2476.3,
    "F280N":2832.9,
    "F343N":3435.1,
    "F373N":3730.2,
    "FQ378N":3792.4,
    "FQ387N":3873.7,
    "F395N":3955.2,
    "FQ436N": 4367.2,
    "FQ437N": 4371,
    "F469N": 4688.1,
    "F487N": 4871.4,
    "FQ492N": 4933.4,
    "F502N": 5009.6,
    "FQ508N": 5091,
    "FQ575N": 5757.7,
    "FQ619N": 6198.5,
    "F631N": 6304.3,
    "FQ634N": 6349.2,
    "F645N": 6453.6,
    "F656N": 6561.4,
    "F657N": 6566.6,
    "F658N": 6584,
    "F665N": 6655.9,
    "FQ672N": 6716.4,
    "F673N": 6765.9,
    "FQ674N": 6730.7,
    "F680N": 6877.6,
    "FQ727N": 7275.2,
    "FQ750N": 7502.5,
    "FQ889N": 8892.2,
    "FQ906N": 9057.8,
    "FQ924N": 9247.6,
    "FQ937N": 9372.4,
}
