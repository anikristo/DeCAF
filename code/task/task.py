# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np

from utils import decaf
import dataset

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
        
        
class DomainAdaptationTask(Task):

    def __init__(self, origin_domain, target_domain, combo):
        super(DomainAdaptationTask, self).__init__()
        # define dataset. domains = {0: "amazon", 1: "dslr", 2: "webcam"}
        
        print("Transfering from", origin_domain, "to", target_domain)
        if combo == "S":
            self.origin_domain_dataset = dataset.OfficeDataset(domain = origin_domain, split=[1,0,0])
            self.target_domain_dataset = dataset.OfficeDataset(domain = target_domain, split=[0, 0, 1])
        else:
            self.origin_domain_dataset = dataset.OfficeDataset(domain = origin_domain, split=[1,0,0])
            self.target_domain_dataset = dataset.OfficeDataset(domain = target_domain, split=[0.3, 0, 0.7])
            
        self.combo = combo

        # define model
        from sklearn import linear_model
        self.model = linear_model.SGDClassifier()

    def train(self):
        
        if "S" in self.combo:
            idx = 1
            for (train_data, train_labels) in self.origin_domain_dataset.get_train_batch_iter():
                print("Origin Domain Train", idx)
                train_decaf_data = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: train_data})
                self.model.partial_fit(train_decaf_data, train_labels, classes=self.origin_domain_dataset.get_labels())
                idx += 1
        
        if "T" in self.combo:
            print "Origin Domain Train: done!"
            idx = 1
            for (train_data, train_labels) in self.target_domain_dataset.get_train_batch_iter():
                print("Target Domain Train", idx)
                train_decaf_data = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: train_data})
                self.model.partial_fit(train_decaf_data, train_labels, classes=self.target_domain_dataset.get_labels())
                idx += 1
                
            print "Target Domain Train: done!"
        print 'Train: done!'

    def test(self):
        from sklearn.metrics import accuracy_score
        scores = []
        for (test_data, test_labels) in self.target_domain_dataset.get_test_batch_iter():
            test_decaf_data = self.sess.run(self.decaf_tensor, feed_dict={self.input_tensor: test_data})
            test_predictions = self.model.predict(test_decaf_data)
            scores.append(accuracy_score(test_labels, test_predictions))
        print 'Accuracy: {}'.format(np.average(scores))
        print 'Test: done!'
        
        
class SceneObjectRecognitionTask(Task):

    def __init__(self):
        super(SceneObjectRecognitionTask, self).__init__()
        # define dataset
        self.dataset = dataset.SUN397Dataset()

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