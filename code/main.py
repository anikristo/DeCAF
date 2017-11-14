# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.misc import imread
import tensorflow as tf

import decaf

def read_image(image_path):
    image = (imread(image_path)[:, :, :3]).astype(np.float32)
    image = image - np.mean(image)
    image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]

    return image

if __name__ == "__main__":
    x_dim = (227, 227, 3)
    x = tf.placeholder(tf.float32, (None,) + x_dim)
    decaf_tensor = decaf.get_decaf_tensor(x)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start = time.time()
    image = read_image("laska.png")
    output = sess.run(decaf_tensor, feed_dict={x: [image]})
    print output
    print output.shape