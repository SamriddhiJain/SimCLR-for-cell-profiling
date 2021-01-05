import tensorflow as tf
import numpy as np
import pandas as pd

from train import train_model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

BATCH_SIZE = 128

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


def get_splitted_dataset(data_path, validation_frac=0.3):
    dataframe = pd.read_csv(data_path + "sc-metadata.csv")
    labels = dataframe['Target']
    images = [data_path + path for path in dataframe['Image_Name']]
    print("Number of images:", len(images))

    features_dataset = tf.data.Dataset.from_tensor_slices(images)
    features_dataset = (
        features_dataset
        .map(parse_cell_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    full_ds = tf.data.Dataset.zip((features_dataset, labels_dataset))
    full_ds = (
        full_ds
        .shuffle(1024)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Train-test split
    train_ds = full_ds.take(int((1 - validation_frac) * len(full_ds)))
    val_ds = full_ds.skip(int((1 - validation_frac) * len(full_ds)))
    return train_ds, val_ds


def get_separated_dataset(train_path, validation_path):
    # Train
    train_dataframe = pd.read_csv(train_path + "sc-metadata.csv")
    train_labels = train_dataframe['Target']
    train_images = [train_path + path for path in train_dataframe['Image_Name']]
    print("Number of train images:", len(train_images))

    features_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    features_dataset = (
        features_dataset
        .map(parse_cell_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
    )
    labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)

    train_ds = tf.data.Dataset.zip((features_dataset, labels_dataset))
    train_ds = (
        train_ds
        .shuffle(20480)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Validation
    val_dataframe = pd.read_csv(validation_path + "sc-metadata.csv")
    val_labels = val_dataframe['Target']
    val_images = [validation_path + path for path in val_dataframe['Image_Name']]
    print("Number of validation images:", len(val_images))

    features_dataset = tf.data.Dataset.from_tensor_slices(val_images)
    features_dataset = (
        features_dataset
        .map(parse_cell_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
    )
    labels_dataset = tf.data.Dataset.from_tensor_slices(val_labels)

    val_ds = tf.data.Dataset.zip((features_dataset, labels_dataset))
    val_ds = (
        val_ds
            .shuffle(10240)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return train_ds, val_ds


# # Train-test split data
# data_path = "../single-cell-sample/"
# train_ds, val_ds = get_splitted_dataset(data_path)

# Separated data
train_path = "/cluster/scratch/agorji/cropped/single-cell-train/"
validation_path = "/cluster/scratch/agorji/cropped/single-cell-test/"

train_ds, val_ds = get_separated_dataset(train_path, validation_path)

train_model(train_ds, val_ds, epochs=300, checkpoint_path="splitted_simclr_epoch{}.h5")
