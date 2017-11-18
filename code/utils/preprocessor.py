__author__ = "Ani Kristo"
__credits__ = ["Ani Kristo"]  # TODO Add other contributors
__status__ = "Development"

'''
    Preprocessor class for the datasets. Typical usage:
    <code>
        preprocessor = Preprocessor(Dataset.DATASET_CALTECH_101)
        for batch in preprocessor.next_batch():
            # Do something useful
            print (batch)
    </code>
'''

import os
import random

import numpy as np
from enum import Enum
from scipy.misc import imread


class Dataset(Enum):
    DATASET_ILSVRC_2012 = 0,
    DATASET_OFFICE = 1,
    DATASET_CALTECH_101 = 2,
    DATASET_BIRDS = 3,
    DATASET_SUN_397 = 4


class Preprocessor:
    def __init__(self, dataset_choice, dataset_dir="../datasets", batch_sz=20):
        """
        Class initializer that enforces the specification of the dataset to be used and allows for custom declaration
        of the dataset directory and batch size for the training and testing phase of the models.

        :param dataset_choice: Specify the behavior of the preprocess with respect with the dataset in context.
        :type dataset_choice: Dataset
        :param dataset_dir: The path to the root of all dataset directories
        :type dataset_dir: str
        :param batch_sz: The size of features and labels to be returned in each batch
        :type batch_sz: int
        """

        # Validation checks for the dataset choice
        if not isinstance(dataset_choice, Dataset):
            raise Exception(
                "Invalid argument for dataset_choice: " + str(dataset_choice) + ". Use the Dataset enumeration.")

        self._dataset_choice = dataset_choice
        self._dataset_dir = dataset_dir
        self._batch_sz = batch_sz

    def next_batch(self):
        """
        :returns: The next batch of features and the respective labels.
        """

        if self._dataset_choice == Dataset.DATASET_ILSVRC_2012:
            raise NotImplementedError  # TODO
        elif self._dataset_choice == Dataset.DATASET_OFFICE:
            raise NotImplementedError  # TODO
        elif self._dataset_choice == Dataset.DATASET_CALTECH_101:
            return self._next_batch_caltech_101(dataset_dir=self._dataset_dir)
        elif self._dataset_choice == Dataset.DATASET_BIRDS:
            raise NotImplementedError  # TODO
        elif self._dataset_choice == Dataset.DATASET_SUN_397:
            raise NotImplementedError  # TODO
        else:
            raise Exception(
                "Invalid argument for dataset_choice: " + self._dataset_choice + ". Use the Dataset enumeration.")

    def _next_batch_caltech_101(self, dataset_dir):
        """
        :return: A batch of features and labels from the Caltech-101 Dataset
        """

        caltech_101_path = dataset_dir + "/Caltech-101"

        labels = []
        for subdir in os.listdir(caltech_101_path):
            if os.path.isdir(caltech_101_path + "/" + subdir):
                labels.append(subdir)

        data = []  # List of tuples of the form (label, image_path)
        for lbl in labels:
            class_path = caltech_101_path + "/" + lbl

            # Collect image paths
            for img_filename in os.listdir(class_path):
                data.append(
                    (lbl, class_path + "/" + img_filename))  # Appends a tuple of (label, image_path)

        # Shuffle data
        data_sz = len(data)
        random.shuffle(data)

        # Split data into training, validation and test set in 60-20-20 ratio
        train_sz = int(data_sz * .6)
        validation_sz = int(data_sz * .2)
        test_sz = data_sz - (train_sz + validation_sz)

        train_set = data[: train_sz]
        validation_set = data[train_sz: train_sz + validation_sz]
        test_set = data[train_sz + validation_sz:]

        for idx in range(0, data_sz, self._batch_sz):
            if idx + self._batch_sz <= data_sz:
                train_batch = np.asarray(
                    list(map(lambda x: self._read_image(x[1]), train_set[idx: idx + self._batch_sz])))
                validation_batch = np.asarray(
                    list(map(lambda x: self._read_image(x[1]), validation_set[idx: idx + self._batch_sz])))
                test_batch = np.asarray(
                    list(map(lambda x: self._read_image(x[1]), test_set[idx: idx + self._batch_sz])))

                res = np.stack((train_batch, validation_batch, test_batch), axis=0)  # First dimension: Batch Size
                yield res

    def _read_image(self, path):
        """
        Read image wrt to AlexNet specifications

        :param path: The file path to the image to be read
        :return: A normalized 3D matrix representation of the image in BGR space
        """

        image = imread(path)
        is_grayscale = image.ndim == 2

        if is_grayscale:
            image = np.stack((
                image,
                np.zeros(shape=image.shape, dtype=np.float32),
                np.zeros(shape=image.shape, dtype=np.float32)),
                axis=2)  # Add zeros for the third channel

        image = image.astype(np.float32)
        image -= np.mean(image)  # Normalize values

        if not is_grayscale:  # NOTE: Switching doesn't make sense for grayscale images
            image[:, :, 2], image[:, :, 0] = image[:, :, 0], image[:, :, 2]  # Switch to BGR space

        return image
