# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import numpy as np
from scipy.misc import imread, imresize

class Dataset(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_train_batch_iter(self):
        pass

    @abstractmethod
    def get_validation_batch_iter(self):
        pass

    @abstractmethod
    def get_test_batch_iter(self):
        pass

    @abstractmethod
    def get_labels(self):
        pass

    def _read_image(self, image_path):
        image = imread(image_path)
        image = imresize(image, size=(227, 227))
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