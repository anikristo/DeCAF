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

    @staticmethod
    def _read_image(image_path, bounding_box=None):
        image = imread(image_path)

        is_grayscale = image.ndim == 2
        if is_grayscale:
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        has_transparency = image.shape[2] == 4
        if has_transparency:
            image = image[:, :, :3]  # Remove the alpha layer

        if bounding_box is not None:
            image = Dataset._crop_image(image, bounding_box)

        image = imresize(image, size=(227, 227))  # TODO maybe use crop

        image = image.astype(np.float32)
        image -= np.mean(image)  # Normalize values

        if not is_grayscale:  # NOTE: Switching doesn't make sense for grayscale images
            image[:, :, 2], image[:, :, 0] = image[:, :, 0], image[:, :, 2]  # BGR space for compatibility with OpenCV

        return image

    @staticmethod
    def _crop_image(image, box):
        imheight, imwidth = image.shape[:2]
        x, y, width, height = box
        centerx = x + width / 2.
        centery = y + height / 2.
        xoffset = width / 2.
        yoffset = height / 2.
        xmin = max(int(centerx - xoffset + 0.5), 0)
        ymin = max(int(centery - yoffset + 0.5), 0)
        xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
        ymax = min(int(centery + yoffset + 0.5), imheight - 1)
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            raise ValueError("The cropped bounding box has size 0.")
        return image[ymin:ymax, xmin:xmax]

        # TODO Crop to dimensions without imresize
