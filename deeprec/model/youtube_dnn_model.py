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

    _nce_weight: VariableV1
    _user_vector: object
    _loss: object

    def __int__(self, item_size, item_embedding_size, layers_units: list[int], keep_prob, num_sampled=64):
        self._item_size = item_size
        self._item_embedding_size = item_embedding_size
        self._layers_units = layers_units
        assert len(self._layers_units) == 3
        self._keep_prob = keep_prob
        self._num_sampled = num_sampled
        self._input_context = None
        self._dense_label = None
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


class YouTubeDnnModelLastViewsSample(YouTubeDnnBaseModel):

    def __init__(self, item_size, item_embedding_size,
                 third_category_size, third_categroy_embeding_size,
                 tag_size, tag_embedding_size,
                 kis_size, kis_embedding_size,
                 album_size, album_embedding_size,
                 author_size, author_embedding_size,
                 layers_units: list[int], keep_prob, lr, epoch):

        self._third_category_size = third_category_size
        self._third_category_embedding_size = third_categroy_embeding_size
        self._tag_size = tag_size
        self._tag_embedding_size = tag_embedding_size
        self._kis_size = kis_size
        self._kis_embedding_size = kis_embedding_size
        self._album_size = album_size
        self._album_embedding_size = album_embedding_size
        self._author_size = author_size
        self._author_embedding_size = author_embedding_size
        self._lr = lr
        self._epoch = epoch
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        super(YouTubeDnnModelLastViewsSample, self).__init__(item_size, item_embedding_size, layers_units, keep_prob)

    def _parse_function(self, serialize_String):
        feature_description = {
            "uid": tf.VarLenFeature(dtype=tf.string),
            "history_vid": tf.VarLenFeature(dtype=tf.int64),
            "history_category": tf.VarLenFeature(dtype=tf.int64),
            "history_tag": tf.VarLenFeature(dtype=tf.int64),
            "history_kis": tf.VarLenFeature(dtype=tf.int64),
            "history_album": tf.VarLenFeature(dtype=tf.int64),
            "history_author": tf.VarLenFeature(dtype=tf.int64),

            "last_vid": tf.VarLenFeature(dtype=tf.int64),
            "last_category": tf.VarLenFeature(dtype=tf.int64),
            "last_tag": tf.VarLenFeature(dtype=tf.int64),
            "last_kis": tf.VarLenFeature(dtype=tf.int64),
            "last_album": tf.VarLenFeature(dtype=tf.int64),
            "last_author": tf.VarLenFeature(dtype=tf.int64),

            "expose_vid": tf.VarLenFeature(dtype=tf.int64),
            "expose_category": tf.VarLenFeature(dtype=tf.int64),
            "expose_tag": tf.VarLenFeature(dtype=tf.int64),
            "expose_kis": tf.VarLenFeature(dtype=tf.int64),
            "expose_album": tf.VarLenFeature(dtype=tf.int64),
            "expose_author": tf.VarLenFeature(dtype=tf.int64),
            "label_sample": tf.VarLenFeature(dtype=tf.float32),
        }

        features = tf.io.parse_single_example(serialized=serialize_String, features=feature_description)

        # 用户uid
        uid = features["uid"]

        # 特征的相关描述
        history_vid = features["history_vid"]
        history_category = features["history_category"]
        history_tag = features["history_tag"]
        history_kis = features["history_kis"]
        history_album = features["history_album"]
        history_author = features["history_author"]

        last_vid = features["last_vid"]
        last_category = features["last_category"]
        last_tag = features["last_tag"]
        last_kis = features["last_kis"]
        last_album = features["last_album"]
        last_author = features["last_author"]

        expose_vid = features["expose_vid"]
        expose_category = features["expose_category"]
        expose_tag = features["expose_tag"]
        expose_kis = features["expose_kis"]
        expose_album = features["expose_album"]
        expose_author = features["expose_author"]

        label_sample = features["label_sample"]

        return (uid, history_vid, history_category, history_tag, history_kis, history_album, history_author,
                last_vid, last_category, last_tag, last_kis, last_album, last_author,
                expose_vid, expose_category, expose_tag, expose_kis, expose_album, expose_author, label_sample)

    def create_data(self):
        with tf.name_scope("create_data"):
            self.dataset_record = tf.data.TFRecordDataset(self._input).prefetch(30000)
            self.parse_dataset = self.dataset_record.map(self.__parse_function).batch(512)
            self.data_iterator = self.parse_dataset.make_initializable_iterator()

            (self._uid, self._history_vid, self._history_category,
             self._history_tag, self._history_kis, self._history_album,
             self._history_author, self._last_vid, self._last_category,
             self._last_tag, self._last_album, self._last_author,
             self._expose_vid, self._expose_category, self._expose_tag,
             self._expose_kis, self._expose_album, self._expose_author, self._label_sample) = self.data_iterator.get_next()

    def _create_embedding(self):
        with tf.name_scope("embed"):
            self._item_matrix = tf.Variable(tf.random_uniform([self._item_size,
                                                               self._item_embedding_size], -1.0, 1.0),
                                            name="item_matrix")

            self._category_matrix = tf.Variable(tf.random_uniform([self._third_category_size,
                                                                   self._third_category_embedding_size], -1.0, 1.0),
                                                name="category_matrix")

            self._tag_matrix = tf.Variable(tf.random_uniform([self._tag_size,
                                                              self._tag_embedding_size], -1.0, 1.0),
                                           name="tag_matrix")

            self._kis_matrix = tf.Variable(tf.random_uniform([self._kis_size,
                                                              self._kis_embedding_size], -1.0, 1.0),
                                           name="kis_matrix")

            self._album_matrix = tf.Variable(tf.random_uniform([self._album_size,
                                                                self._album_embedding_size], -1.0, 1.0),
                                             name="album_matrix")

            self._author_matrix = tf.Variable(tf.random_uniform([self._author_size,
                                                                self._author_embedding_size], -1.0, 1.0),
                                             name="album_matrix")

    def _create_input_context(self):
        with tf.name_scope("create_input_context"):
            self._history_context = tf.nn.embedding_lookup_sparse(self._item_matrix,
                                                                  self._history_vid,
                                                                  combiner="mean",
                                                                  sp_weights=None,
                                                                  name="history_context")

            self._category_context = tf.nn.embedding_lookup_sparse(self._category_matrix,
                                                                   self._history_category,
                                                                   combiner="mean",
                                                                   sp_weights=None,
                                                                   name="category_context")

            self._tag_context = tf.nn.embedding_lookup_sparse(self._tag_matrix,
                                                             self._history_tag,
                                                             combiner="mean",
                                                             sp_weights=None,
                                                             name="sc_context")

            self._kis_context = tf.nn.embedding_lookup_sparse(self._kis_matrix,
                                                              self._history_kis,
                                                              combiner="mean",
                                                              sp_weights=None,
                                                              name="tc_context")

            self._album_context = tf.nn.embedding_lookup_sparse(self._album_matrix,
                                                                self._history_album,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="album_context")

            self._author_context = tf.nn.embedding_lookup_sparse(self._author_matrix,
                                                                self._history_author,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="author_context")
    def _create_last_view_context(self):

        with tf.name_scope("create_input_context"):
            self._last_vid_context = tf.nn.embedding_lookup_sparse(self._item_matrix,
                                                                   self._history_vid,
                                                                   combiner="mean",
                                                                   sp_weights=None,
                                                                   name="last_vid_context")

            self._last_category_context = tf.nn.embedding_lookup_sparse(self._category_matrix,
                                                                   self._history_category,
                                                                   combiner="mean",
                                                                   sp_weights=None,
                                                                   name="last_category_context")

            self._last_tag_contex = tf.nn.embedding_lookup_sparse(self._tag_matrix,
                                                             self._history_tag,
                                                             combiner="mean",
                                                             sp_weights=None,
                                                             name="last_tag_context")

            self._last_kis_context = tf.nn.embedding_lookup_sparse(self._kis_matrix,
                                                              self._history_kis,
                                                              combiner="mean",
                                                              sp_weights=None,
                                                              name="last_kis_context")

            self._last_album_context = tf.nn.embedding_lookup_sparse(self._album_matrix,
                                                                self._history_album,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="last_album_context")

            self._last_author_context = tf.nn.embedding_lookup_sparse(self._author_matrix,
                                                                self._history_author,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="last_author_context")


    def creat_input_context(self):
        with tf.name_scope("create_input_context"):
            self._input_context = tf.concat([self._history_context, self._category_context, self._tag_context,
                                             self._kis_context, self._album_context, self._author_context,
                                             self._last_vid_context, self._last_category_context, self._last_tag_contex,
                                             self._last_album_context, self._last_author_context])

    def _create_smaple_conext(self):
        with tf.name_scope("sample_context"):
            self._expose_vid_dense = tf.sparse_tensor_to_dense(self._expose_vid)
            self._expose_category_dense = tf.sparse_tensor_to_dense(self._expose_category)
            self._expose_tag_dense = tf.sparse_tensor_to_dense(self._expose_tag)
            self._expose_kis_dense = tf.sparse_tensor_to_dense(self._expose_kis)
            self._expose_album_dense = tf.sparse_tensor_to_dense(self._expose_album)
            self._expose_author_dense = tf.sparse_tensor_to_dense(self._expose_author_dense)

            self._dense_label = tf.sparse_tensor_to_dense(self._label_sample)

            self._expose_vid_context = tf.nn.embedding_lookup(self._item_matrix, self._expose_vid_dense)
            self._expose_category_context = tf.nn.embedding_lookup(self._category_matrix, self._expose_category_dense)
            self._expose_tag_context = tf.nn.embedding_lookup(self._tag_matrix, self._expose_tag_dense)
            self._expose_kis_context = tf.nn.embedding_lookup(self._kis_matrix, self._expose_kis_dense)
            self._expose_album_context = tf.nn.embedding_lookup(self._album_matrix, self._expose_album_dense)
            self._expose_author_context = tf.nn.embedding_lookup(self._author_matrix, self._expose_author_dense)

            self._sample_context = tf.concat([self._expose_vid_context,
                                              self._expose_category_context,
                                              self._expose_tag_context,
                                              self._expose_kis_context,
                                              self._expose_album_context,
                                              self._expose_author_context], -1)

    def _create_sample_loss(self):
        with tf.name_scope("sample_loss"):
            normal_sample_matrix = tf.nn.l2_normalize(self._sample_context, 1)
            self._distance = tf.squeeze(
                tf.matmul(tf.transpose(self._user_vector, perm=[0, 1, 2]),
                          tf.transpose(normal_sample_matrix, perm=[0, 2, 1]), name="distance"),
                axis=1)

            self._softmax_dist = tf.nn.softmax_cross_entropy_with_logits(labels=self._dense_label,
                                                                         logits=self._distance)
            self._sample_loss = tf.reduce_mean(self._softmax_dist)

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self._optimizer = tf.train.GradientDescentOptimizer(self._lr).minimize(self._sample_loss,
                                                                                   global_step=self.global_step)

    def _carete_candidate_embedding(self):
        with tf.name_scope("candidate"):
            self._vid_dense = tf.placeholder(tf.int32, shape=[None], name='vid_dense')
            self._category_dense = tf.placeholder(tf.int32, shape=[None], name='category_dense')
            self._tag_dense = tf.placeholder(tf.int32, shape=[None], name='tag_dense')
            self._kis_dense = tf.placeholder(tf.int32, shape=[None], name='kis_dense')
            self._album_dense = tf.placeholder(tf.int32, shape=[None], name='album_dense')

            self._all_vid_context = tf.nn.embedding_lookup(self._item_matrix, self._vid_dense)
            self._all_category_context = tf.nn.embedding_lookup(self._category_matrix, self._category_dense)
            self._all_tag_context = tf.nn.embedding_lookup(self._tag_matrix, self._tag_dense)
            self._all_kis_context = tf.nn.embedding_lookup(self._kis_matrix, self._kis_dense)
            self._all_album_context = tf.nn.embedding_lookup(self._album_matrix, self._album_dense)

            self._all_embedd_context = tf.concat([self._all_vid_context,
                                                  self._all_category_context,
                                                  self._all_tag_context,
                                                  self._all_kis_context,
                                                  self._all_album_context], -1)
