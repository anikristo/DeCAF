#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:49:42 2017

@author: aalto
"""

# -*- coding: utf-8 -*-

import os
import random

import numpy as np

from dataset import Dataset


class SUN397Dataset(Dataset):

    def __init__(self, batch_size=20, dataset_dir=os.path.join("../datasets", "SUN397/a")):
        self.batch_size = batch_size

        labels = []
        for subdir in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, subdir)):
                labels.append(subdir)

        label_encodings = dict(map(lambda t: (t[1], t[0]), enumerate(set(labels))))

        # List of tuples of the form (label, image_path)

        train_set = []
        test_set = []
        for lbl in labels:
            class_path = os.path.join(dataset_dir, lbl)
            data_buffer = []
            # Collect image paths
            for img_filename in os.listdir(class_path):
                if len(data_buffer) < 100:
                    data_buffer.append(
                        (label_encodings[lbl],
                         os.path.join(class_path, img_filename)))  # Appends a tuple of (label, image_path)
                else:
                    random.shuffle(data_buffer)
                    train_set.extend(data_buffer[:50])
                    test_set.extend(data_buffer[50:])
                    break

        random.shuffle(train_set)
        random.shuffle(test_set)

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
        self.test_batch_iter = get_batch(test_set)
        self.labels = label_encodings.values()

    def get_train_batch_iter(self):
        return self.train_batch_iter

    def get_validation_batch_iter(self):
        pass

    def get_test_batch_iter(self):
        return self.test_batch_iter

    def get_labels(self):
        return self.labels
