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
                        (label_encodings[lbl], os.path.join(class_path, img_filename)))  # Appends a tuple of (label, image_path)
                else:
                    random.shuffle(data_buffer)
                    train_set.extend(data_buffer[:50])
                    test_set.extend(data_buffer[50:])
                    break
                    
                    
        random.shuffle(train_set)
        random.shuffle(test_set)
                
# =============================================================================
#             for img_filename in os.listdir(class_path):
#                 image_path = os.path.join(class_path, img_filename)
#                 im=  imread(image_path)
#                 #print im.shape, image_path
#                 if len(im.shape) == 2 or im.shape[2] == 3:
#                     data.append(
#                         (label_encodings[lbl], image_path))  # Appends a tuple of (label, image_path)
#                 else:
#                     os.rename(image_path, os.path.join("/home/aalto/2470/dl/CS2470-project/datasets/SUN397/junk", img_filename))
# 
#         # Shuffle data
#         data_size = len(data)
#         random.shuffle(data)
# 
#         # Split data into training, validation and test set in 60-20-20 ratio
#         train_size = int(data_size * .7)
#         validation_size = int(data_size * .2)
#         test_size = data_size - (train_size + validation_size)
# 
#         train_set = data[:train_size]
#         validation_set = data[train_size:train_size + validation_size]
#         test_set = data[train_size + validation_size:]
# =============================================================================

        def get_batch(data_set):
            data_size = len(data_set)
            for idx in range(0, data_size, self.batch_size):
                if idx + self.batch_size <= data_size:
                    data_batch = np.asarray(
                    list(map(lambda x: self._read_image(x[1]), data_set[idx: idx + self.batch_size])))
                    labels_batch = np.asarray(
                    list(map(lambda x: x[0], data_set[idx: idx + self.batch_size])))
                    yield (data_batch, labels_batch)

        self.train_batch_iter = get_batch(train_set)
        #self.validation_batch_iter = get_batch(validation_set)
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