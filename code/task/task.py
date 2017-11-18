# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from abc import abstractmethod, ABCMeta
import tensorflow as tf

from utils import decaf
from utils import preprocessor

class Task(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.input_tensor = tf.placeholder(tf.float32, (None, 227, 227, 3))
        self.decaf_tensor = decaf.get_decaf_tensor(self.input_tensor)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

class ObjectRecognitionTask(Task):

    def __init__(self):
        super(ObjectRecognitionTask, self).__init__()

    def train(self):
        # output = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: [image]})

        print 'train'

    def test(self):
        print 'test'
