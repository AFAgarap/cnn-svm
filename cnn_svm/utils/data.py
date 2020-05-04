# Convolutional Neural Network with Support Vector Machine
# Copyright (C) 2017-2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def load_tfds(
    name: str = "mnist"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a data set from `tfds`.

    Parameters
    ----------
    name : str
        The name of the TensorFlow data set to load.

    Returns
    -------
    train_features : np.ndarray
        The train features.
    test_features : np.ndarray
        The test features.
    train_labels : np.ndarray
        The train labels.
    test_labels : np.ndarray
        The test labels.
    """
    train_dataset = tfds.load(name=name, split=tfds.Split.TRAIN, batch_size=-1)
    train_dataset = tfds.as_numpy(train_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features.reshape(-1, np.prod(train_features.shape[1:]))
    train_features = train_features / 255.0

    test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
    test_dataset = tfds.as_numpy(test_dataset)

    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    test_features = test_features.astype("float32")
    test_features = test_features.reshape(-1, np.prod(test_features.shape[1:]))
    test_features = test_features / 255.0

    return train_features, test_features, train_labels, test_labels


def create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    as_supervised: bool = True,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` object from a pair of
    `features` and `labels` or `features` alone.

    Parameters
    ----------
    features : np.ndarray
        The features matrix.
    labels : np.ndarray
        The labels matrix.
    batch_size : int
        The mini-batch size.
    as_supervised : bool
        Boolean whether to load the dataset as supervised or not.

    Returns
    -------
    dataset : tf.data.Dataset
        The dataset pipeline object, ready for model usage.
    """
    if as_supervised:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(features.shape[1])
    return dataset
