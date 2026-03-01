"""FER2013 data loading and augmentation utilities."""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 48
NUM_CLASSES = 7


def parse_pixels(pixel_string):
    """Convert a CSV pixel string to a 48x48x1 float32 array normalized to [0, 1]."""
    return (
        np.array([int(p) for p in pixel_string.split()], dtype=np.float32)
        .reshape(IMG_SIZE, IMG_SIZE, 1)
        / 255.0
    )


def load_fer2013(csv_path):
    """Load FER2013 CSV and split by Usage column.

    Returns a dict with keys: X_train, y_train, y_train_oh,
    X_val, y_val, y_val_oh, X_test, y_test, y_test_oh.
    """
    df = pd.read_csv(csv_path)

    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"] == "PublicTest"]
    test_df = df[df["Usage"] == "PrivateTest"]

    X_train = np.array([parse_pixels(p) for p in train_df["pixels"]])
    y_train = train_df["emotion"].values
    X_val = np.array([parse_pixels(p) for p in val_df["pixels"]])
    y_val = val_df["emotion"].values
    X_test = np.array([parse_pixels(p) for p in test_df["pixels"]])
    y_test = test_df["emotion"].values

    return {
        "X_train": X_train, "y_train": y_train,
        "y_train_oh": to_categorical(y_train, NUM_CLASSES),
        "X_val": X_val, "y_val": y_val,
        "y_val_oh": to_categorical(y_val, NUM_CLASSES),
        "X_test": X_test, "y_test": y_test,
        "y_test_oh": to_categorical(y_test, NUM_CLASSES),
    }


def get_train_datagen():
    """Return ImageDataGenerator with V3 augmentation settings."""
    return ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
    )
