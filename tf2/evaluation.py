import tensorflow as tf

from feature_eval.random_forest_classifier import RFClassifier

print(tf.__version__)

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from imutils import paths
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pickle
import h5py

from losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
import helpers

@tf.function
def parse_cell_images(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string, channels=1)
    image = tf.squeeze(image)
    image = tf.split(image, 3, axis=1)
    image = tf.transpose(image, [1,2,0])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[96, 96])

    return image


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

train_images = list(paths.list_images("../single-cell-sample/"))
print("Number of images:", len(train_images))

BATCH_SIZE = 128

train_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_ds = (
    train_ds
    .map(parse_cell_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .shuffle(1024)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


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


# Mask to remove positive examples from the batch of negative samples
negative_mask = helpers.get_negative_mask(BATCH_SIZE)


@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_simclr(model, dataset, optimizer, criterion,
                 temperature=0.1, epochs=100, start=0):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(start, epochs)):
        for image_batch in dataset:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            del a
            del b
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        #wandb.log({"nt_xentloss": np.mean(step_wise_loss)})

        if epoch % 10 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        if epoch % 50 == 0:
            print("Saving checkpoint")
            model.save_weights("resnet_simclr_epoch{}.h5".format(epoch))
            with open("simclr_losses_epoch{}.pkl".format(epoch), "wb") as fp:
                pickle.dump(epoch_wise_loss, fp)

    return epoch_wise_loss, model


def validate(model, dataset):
    embeddings = []
    for image_batch in dataset:
        embedding_batch = model(image_batch)
        print(embedding_batch)
        print(embedding_batch.shape)
        embeddings.extend(embedding_batch)
    classifier = RFClassifier(model)
    train_original, validate_original = self.eval_dataset.get_data_loaders()
    print("RF classifier training started.")
    classifier.train(train_original)
    print("Training classifier done.")
    score_eval = classifier.test(validate_original)
    print(f"Classifier accuracy {score_eval}")


criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.SUM)
decay_steps = 1000
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

epoch_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
model = get_resnet_simclr(256, 128, 50)
model.load_weights("resnet_simclr_epoch{}.h5".format(start))

rep = model(xis)
epoch_wise_loss, resnet_simclr  = train_simclr(model, train_ds, optimizer, criterion,
                                               temperature=0.1, epochs=500, start=start)

import datetime
filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "resnet_simclr.h5"
model.save_weights(filename)
