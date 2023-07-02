# Author: Jacob Dawson
#
# This is the main file we will be using to process the photos; we train our
# model here. I want to note that most of the work in this repo is a copy of
# the work I've done on the JunoGAN project, another repo doing a similar thing
# to images by NASA's Juno spacecraft. Here, we'll be training a GAN to
# colorize/edit JWST images to make them nice and pretty!

# IMPORTS
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import imageio

# our imports:
from constants import *
from architecture import *
from datasetGenerator import *  # also imports the datasets themselves

physical_devices = tf.config.experimental.list_physical_devices("GPU")
num_gpus = len(physical_devices)
# if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

keras.mixed_precision.set_global_policy("mixed_float16")
tf.keras.utils.set_random_seed(seed)

# these are declared in architecture.py
generator = gen()
discriminator = dis()


# let's print some useful info here
print("\n")
print("######################################################################")
print("\n")
print("Architecture:\n")
print("\n")
print("\n")
print("Image size:", image_size)
print("Dataset size:", combinedCardinality)
print("Target images:", officialCardinality)
print("Raw images:", rawCardinality)
print("Batch size:", batch_size)
print("Weight of content loss:", content_lambda)
print("Weight of WGAN loss:", wgan_lambda)
print("Value of hyperparameter chi:", chi)
print("g learning rate", gen_learn_rate)
print("d learning rate", dis_learn_rate)
print("Intended number of epochs:", epochs)
print("Number of GPUs we're running on:", num_gpus)
print("\n")
print("######################################################################")
print("\n")


def content_loss(fake, real):
    f = tf.cast(fake, tf.float32)
    r = tf.cast(real, tf.float32)
    ssim = chi * (1.0 - tf.experimental.numpy.mean(tf.image.ssim(f, r, 1.0)))
    l1 = (1.0 - chi) * tf.norm(
        (f / (batch_size * 255.0)) - (r / (batch_size * 255.0))
    )
    return tf.cast(ssim, tf.float16) + tf.cast(l1, tf.float16)


# and here we create the ConditionalGAN itself. Exciting!
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.dis_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property  # no idea what this does
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]

    def compile(
        self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, run_eagerly
    ):
        super(ConditionalGAN, self).compile(run_eagerly=run_eagerly)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        raw_img_batch, user_img_batch = data

        # generate labels for the real and fake images
        batch_size = tf.shape(raw_img_batch)[0]
        true_image_labels = -tf.cast(tf.ones((batch_size, 1)), tf.float16)
        fake_image_labels = tf.cast(tf.ones((batch_size, 1)), tf.float16)
        # REMEMBER: TRUE IMAGES ARE -1, GENERATED IMAGES ARE +1

        # training here:
        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            gen_output = generator(raw_img_batch, training=True)
            disc_real_output = discriminator(user_img_batch, training=True)
            disc_generated_output = discriminator(gen_output, training=True)
            wganLoss = -self.g_loss_fn(fake_image_labels, disc_generated_output)
            wganLoss = (
                tf.convert_to_tensor(wgan_lambda, dtype=tf.float16) * wganLoss
            )
            contentLoss = content_loss(
                tf.concat(
                    (
                        gen_output,
                        tf.zeros(
                            (batch_size, image_size, image_size, 1),
                            dtype=tf.float16,
                        ),
                    ),
                    axis=3,
                ),
                raw_img_batch,
            )
            contentLoss = (
                tf.convert_to_tensor(content_lambda, dtype=tf.float16)
                * contentLoss
            )
            total_g_loss = wganLoss + contentLoss
            d_loss = self.d_loss_fn(
                fake_image_labels, disc_generated_output
            ) - self.d_loss_fn(true_image_labels, disc_real_output)
        grads = gtape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        self.gen_loss_tracker.update_state(total_g_loss)

        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        self.dis_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.dis_loss_tracker.result(),
            "GAN_loss": wganLoss,
            "content_loss": contentLoss,
        }


# okay... let's try to use this thing:
cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator)


def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


cond_gan.compile(
    # d_optimizer = tf.keras.optimizers.RMSprop(learning_rate = dis_learn_rate),
    # g_optimizer = tf.keras.optimizers.RMSprop(learning_rate = gen_learn_rate),
    d_optimizer=tf.keras.optimizers.Adam(
        learning_rate=dis_learn_rate, beta_1=momentum
    ),
    g_optimizer=tf.keras.optimizers.Adam(
        learning_rate=gen_learn_rate, beta_1=momentum
    ),
    d_loss_fn=wasserstein_loss,
    g_loss_fn=wasserstein_loss,
    run_eagerly=True,
)

# only uncomment this code if you have a prepared checkpoint to use for output:
# cond_gan.built=True
# cond_gan.load_weights("ckpts/ckpt60")
# print("Checkpoint loaded, skipping training.")


class EveryKCallback(keras.callbacks.Callback):
    def __init__(self, data, epoch_interval=5):
        self.data = data
        self.epoch_interval = epoch_interval

    def on_epoch_begin(self, epoch, logs=None):
        # please note that the "raw" pngs aren't a great representation of
        # what's in the data, because the "raw" images that the generator
        # receives may have more than the 3 channels allowed by RGB. What gets
        # printed is instead a reduced form; however, the "fake images" shown
        # are the actual outputs, so they represent what the generator is
        # learning.
        if (epoch % self.epoch_interval) == 0:
            random_selection = self.data.take(1)
            raw_images, _ = list(random_selection.as_numpy_iterator())[0]
            raw_image = tf.convert_to_tensor(raw_images[0], dtype=tf.float32)
            fake_image = self.model.generator(
                tf.expand_dims(raw_image, 0), training=False
            )[0]
            raw_image = raw_image.numpy().astype(np.uint8)[:, :, 0:3]
            # ^ reduce to rgb, otherwise it'll be an RGBA formatted image
            fake_image = fake_image.numpy().astype(np.uint8)
            imageio.imwrite(
                checkPointImageDir + str(epoch) + ".png", fake_image
            )
            imageio.imwrite(
                checkPointImageDir + str(epoch) + "raw.png", raw_image
            )

            self.model.save_weights(
                "ckpts/ckpt" + str(epoch), overwrite=True, save_format="h5"
            )
            self.model.generator.save("telescopeGen", overwrite=True)


cond_gan.fit(
    datasets,
    # data is already batched!
    epochs=epochs,
    verbose=1,
    callbacks=[
        EveryKCallback(datasets, epoch_interval=5)
    ],  # custom callbacks here!
    # validation doesnt really apply here?
    shuffle=False,  # shuffling done via dataset api
)

cond_gan.save_weights("ckpts/finished", overwrite=True, save_format="h5")
cond_gan.generator.save("telescopeGen", overwrite=True)
# for good measure, save again once we're done training
