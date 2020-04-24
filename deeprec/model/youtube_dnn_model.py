#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import tensorflow as tf
from tensorflow.python import VariableV1


class FeatureMeta(metaclass=abc.ABCMeta):

    def __init__(self, name, size, embed_size, data_type):
        self.name = name
        self.size = size
        self.embed_size = embed_size
        self.data_type = data_type


class YouTubeDnnBaseModel(object):

    def __int__(self, item_size, item_embedding_size, layers_units: list[int], keep_prob, num_sampled):
        self._item_size = item_size
        self._item_embedding_size = item_embedding_size
        self._layers_units = layers_units
        assert len(self._layers_units) == 3
        self._keep_prob = keep_prob
        self._num_sampled = num_sampled
        self._input_context = None
        self._dense_label = None
        self._user_vector = None
        self._nce_weight = None

    def create_dnn_process(self):
        """
        create tensor flow graph of deep neural network
        :return:
        """
        with tf.name_scope("dnn_process"):
            assert self._input_context is not None
            batch_normal = tf.layers.batch_normalization(inputs=self._input_context, name="batch_normal")
            layers1 = tf.layers.dense(batch_normal, units=self._layers_units[0], activation=tf.nn.relu, name="layer1")
            layers1_dropout = tf.nn.dropout(layers1, keep_prob=self._keep_prob)
            layers2 = tf.layers.dense(layers1_dropout, units=self._layers_units[1], activation=tf.nn.relu,
                                      name="layer2")
            layers2_dropout = tf.nn.dropout(layers2, keep_prob=self._keep_prob)
            layers3 = tf.layers.dense(layers2_dropout, units=self._layers_units[2], activation=tf.nn.relu,
                                      name="layer3")
            self._user_vector = tf.expand_dims(layers3, 1)

    def create_loss(self):
        """
        create random negative sample loss
        :return:
        """
        with tf.name_scope("sample_loss"):
            self._nce_weight = tf.Variable(tf.random_uniform([self._item_size, self._item_embedding_size]), -1.0, 1.0,
                                           name="item_output_matrix")

            self._loss = tf.reduce_mean(tf.nn.nce_loss(weights=self._nce_weight,
                                                       biases=None,
                                                       labels=self._dense_label,
                                                       inputs=self._user_vector,
                                                       num_sampled=self._num_sampled,
                                                       num_classes=self._item_size), name="loss")

    @abc.abstractmethod
    def create_sample_embedding(self):
        """
        create sample embedding of negative sample
        :return:
        """
        pass

    @abc.abstractmethod
    def create_sample_loss(self):
        pass
