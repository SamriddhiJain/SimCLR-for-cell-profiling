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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


def train_validate_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()

    print("Train size:", len(y_train))
    print("Test size:", len(y_test))
    print("RF classifier training started.")
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    rf.fit(X_train, y_train)
    print("Training classifier done.")
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    return rf.score(X_test, y_test)


def extract_representations(model, dataset):
    embeddings = []
    labels = []
    for image_batch, label_batch in dataset:
        embedding_batch = model(image_batch)
        embeddings.extend(embedding_batch.numpy())
        labels.extend(label_batch)
    return np.array(embeddings), np.array(labels)


def validate(model, train_dataset, validation_dataset):
    train_rep, train_labels = extract_representations(model, train_dataset)
    validation_rep, validation_labels = extract_representations(model, validation_dataset)
    return train_validate_rf(train_rep, validation_rep, train_labels, validation_labels)


if __name__ == "__main__":

    # Random seed fixation
    tf.random.set_seed(666)
    np.random.seed(666)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    metadata_path = "../single-cell-sample/sc-metadata.csv"
    dataframe = pd.read_csv(metadata_path)
    labels = dataframe['Target']
    train_images = ["../single-cell-sample/" + path for path in dataframe['Image_Name']]

    print("Number of images:", len(train_images))

    BATCH_SIZE = 128
    VALIDATION_FRAC = 0.3

    features_dataset = tf.data.Dataset.from_tensor_slices(train_images)
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
    train_ds = full_ds.take(int((1 - VALIDATION_FRAC) * len(full_ds)))
    val_ds = full_ds.skip(int((1 - VALIDATION_FRAC) * len(full_ds)))

    epoch_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    model = get_resnet_simclr(256, 128, 50)
    for epoch in tqdm(epoch_list):
        model.load_weights("resnet_simclr_epoch{}.h5".format(epoch))
        score = validate(model, train_ds, val_ds)
        print("Random forest score epoch {epoch}:".format(epoch=epoch), score)
