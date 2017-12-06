# -*- coding: utf-8 -*-

import os
import random

import numpy as np

from dataset import Dataset


class Caltech101Dataset(Dataset):
    def __init__(self, batch_size=20, dataset_dir=os.path.join("../datasets", "Caltech-101")):
        self.batch_size = batch_size

        labels = []
        for subdir in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, subdir)):
                labels.append(subdir)

        label_encodings = dict(map(lambda t: (t[1], t[0]), enumerate(set(labels))))

        data = []  # List of tuples of the form (label, image_path)
        for lbl in labels:
            class_path = os.path.join(dataset_dir, lbl)

            # Collect image paths
            for img_filename in os.listdir(class_path):
                data.append(
                    (label_encodings[lbl],
                     os.path.join(class_path, img_filename)))  # Appends a tuple of (label, image_path)

        # Shuffle data
        data_size = len(data)
        random.shuffle(data)

        # Split data into training, validation and test set in 60-20-20 ratio
        train_size = int(data_size * .7)
        validation_size = int(data_size * .2)
        test_size = data_size - (train_size + validation_size)

        train_set = data[:train_size]
        validation_set = data[train_size:train_size + validation_size]
        test_set = data[train_size + validation_size:]

        def get_batch(data_set):
            data_size = len(data_set)
            for idx in range(0, data_size, self.batch_size):
                if idx + self.batch_size <= data_size:
                    data_batch = np.asarray(
                        list(map(lambda x: Dataset._read_image(x[1]), data_set[idx: idx + self.batch_size])))
                    labels_batch = np.asarray(
                        list(map(lambda x: x[0], data_set[idx: idx + self.batch_size])))
                    yield (data_batch, labels_batch)

        self.train_batch_iter = get_batch(train_set)
        self.validation_batch_iter = get_batch(validation_set)
        self.test_batch_iter = get_batch(test_set)
        self.labels = label_encodings.values()

    def get_train_batch_iter(self):
        return self.train_batch_iter

    def get_validation_batch_iter(self):
        return self.validation_batch_iter

    def get_test_batch_iter(self):
        return self.test_batch_iter

    def get_labels(self):
        return self.labels
