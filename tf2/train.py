import datetime

import tensorflow as tf

from evaluation import validate

print(tf.__version__)

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
import helpers

# Augmentation utilities (differs from the original implementation)
# Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendxi A
# corresponding GitHub: https://github.com/google-research/simclr/)


class CustomAugment(object):
    def __call__(self, sample):
        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)

        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

# Build the augmentation pipeline
data_augmentation = Sequential([Lambda(CustomAugment())])

# Architecture utils
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(96, 96, 3))
    base_model.trainable = True
    inputs = Input((96, 96, 3))
    h = base_model(inputs, training=True)
    h = GlobalAveragePooling2D()(h)

    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr


@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature, batch_size):
    # Mask to remove positive examples from the batch of negative samples
    negative_mask = helpers.get_negative_mask(batch_size)
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(batch_size, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * batch_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_simclr(model, train_ds, val_ds, optimizer, criterion, save_path, batch_size,
                 temperature=0.1, epochs=100, start=0):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(start, epochs)):
        for image_batch, label_batch in train_ds:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature, batch_size)
            del a
            del b
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        #wandb.log({"nt_xentloss": np.mean(step_wise_loss)})

        if (epoch + 1) % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        if (epoch + 1) % 50 == 0:
            print("Saving checkpoint")
            model.save_weights(save_path.format(epoch))
            with open("last_run_losses_epoch{}.pkl".format((epoch + 1)), "wb") as fp:
                pickle.dump(epoch_wise_loss, fp)
            score = validate(model, train_ds, val_ds)
            print("Random forest score epoch {epoch}:".format(epoch=(epoch + 1)), score)

    return epoch_wise_loss, model


def train_model(train_ds, val_ds, epochs, checkpoint_path,
                initial_learning_rate=0.1, start=0, decay_steps=1000, batch_size=128):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction=tf.keras.losses.Reduction.SUM)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    resnet_simclr_2 = get_resnet_simclr(256, 128, 50)
    if start > 0:
        resnet_simclr_2.load_weights(checkpoint_path.format(start))

    epoch_wise_loss, resnet_simclr  = train_simclr(resnet_simclr_2, train_ds, val_ds, optimizer, criterion,
                                                   checkpoint_path, batch_size,
                                                   temperature=0.1, epochs=epochs, start=start)
