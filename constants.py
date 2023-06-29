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
epochs = 100
rawImgDir = "raw_data/"
officialImgDir = "official_images/"

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
