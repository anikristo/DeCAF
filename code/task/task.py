# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np

from utils import decaf
from utils import dataset

class Task(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.sess = tf.Session()
        self.input_tensor = tf.placeholder(tf.float32, (None, 227, 227, 3))
        self.decaf_tensor = decaf.get_decaf_tensor(self.input_tensor)
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

class ObjectRecognitionTask(Task):

    def __init__(self):
        super(ObjectRecognitionTask, self).__init__()
        # define dataset
        self.dataset = dataset.Caltech101Dataset()

        # define model
        from sklearn import linear_model
        self.model = linear_model.SGDClassifier()

    def train(self):
        idx = 1
        for (train_data, train_labels) in self.dataset.get_train_batch_iter():
            print idx
            train_decaf_data = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: train_data})
            self.model.partial_fit(train_decaf_data, train_labels, classes=self.dataset.get_labels())
            idx += 1

        print 'Train: done!'

    def test(self):
        from sklearn.metrics import accuracy_score
        scores = []
        for (test_data, test_labels) in self.dataset.get_test_batch_iter():
            test_decaf_data = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: test_data})
            test_predictions = self.model.predict(test_decaf_data)
            scores.append(accuracy_score(test_labels, test_predictions))
        print 'Accuracy: {}'.format(np.average(scores))
        print 'Test: done!'