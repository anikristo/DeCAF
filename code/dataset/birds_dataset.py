# -*- coding: utf-8 -*-

import os
import random

import numpy as np

from dataset import Dataset


# Code from: https://github.com/Yangqing/decaf/blob/86b39770476ab2865018ca976321ac1e99e286da/decaf/layers/data/cub.py
class BirdsDataset(Dataset):
    def __init__(self, batch_size=20, dataset_dir=os.path.join("../datasets", "Birds")):
        self.batch_size = batch_size

        dataset_dir = os.path.join(dataset_dir, 'CUB_200_2011')

        image_file_names = [os.path.join(dataset_dir, 'images', line.split()[1]) for line in
                            open(os.path.join(dataset_dir, 'images.txt'), 'r')]
        boxes = [line.split()[1:] for line in
                 open(os.path.join(dataset_dir, 'bounding_boxes.txt'), 'r')]
        labels = [int(line.split()[1]) - 1 for line in
                  open(os.path.join(dataset_dir, 'image_class_labels.txt'), 'r')]
        split = [int(line.split()[1]) for line in
                 open(os.path.join(dataset_dir, 'train_test_split.txt'), 'r')]

        # for the boxes, we store them as a numpy array TODO
        boxes = np.array(boxes, dtype=np.float32)
        boxes -= 1  # -> Python indexing

        # Split training and testing set
        target_training = 1
        train_set = [(label, image, box) for image, box, label, val in zip(image_file_names, boxes, labels, split) if
                     val == target_training]

        target_testing = 0
        test_set = [(label, image, box) for image, box, label, val in zip(image_file_names, boxes, labels, split) if
                    val == target_testing]

        # Shuffle
        random.shuffle(train_set)
        random.shuffle(test_set)

        def get_batch(data_set):
            data_size = len(data_set)
            for idx in range(0, data_size, self.batch_size):
                if idx + self.batch_size <= data_size:
                    data_batch = np.asarray(
                        list(map(lambda x: Dataset._read_image(x[1], x[2]), data_set[idx: idx + self.batch_size])))
                    labels_batch = np.asarray(
                        list(map(lambda x: x[0], data_set[idx: idx + self.batch_size])))
                    yield (data_batch, labels_batch)

        self.train_batch_iter = get_batch(train_set)
        self.test_batch_iter = get_batch(test_set)
        self.labels = list(set(labels))

    def get_train_batch_iter(self):
        return self.train_batch_iter

    def get_validation_batch_iter(self):
        return None

    def get_test_batch_iter(self):
        return self.test_batch_iter

    def get_labels(self):
        return self.labels
