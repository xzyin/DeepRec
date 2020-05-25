#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class BaseModel(object):
    def __init__(self, train_path, test_path, candidate_path,
                 item_size, item_esize,
                 c_size, c_esize,
                 t_size, t_esize,
                 k_size, k_esize,
                 a_size, a_esize,
                 u_size, u_esize,
                 model_path, keep_prob, predict_online=True):

        self._train_path = train_path

        self._test_path = test_path

        self._candidate_path = candidate_path

        # 物品的矩阵大小
        self._item_size = item_size
        self._item_esize = item_esize

        # 三级类矩阵大小
        self._c_size = c_size
        self._c_esize = c_esize

        # 标签矩阵大小
        self._t_size = t_size
        self._t_esize = t_esize

        # kis矩阵大小
        self._k_size = k_size
        self._k_esize = k_esize

        # 专辑矩阵大小
        self._a_size = a_size
        self._a_esize = a_esize

        self._u_size = u_size
        self._u_esize = u_esize

        self._model_path = model_path

        self._keep_prob = keep_prob

        self.predict_online = predict_online

        self.batch_size = 300

        if self.predict_online:
            self.batch_size = 1

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def _BytesListFeature(self, x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

    def _Int64ListFeature(self, x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

    def _FloatListFeature(self, x):
        return tf.train.Feature(float_list=tf.train.FloatList(value=x))

    def _FeaturesListFeature(self, x):
        return tf.train.FeatureList(feature=x)

    def __parse_function(self, serialize_String):
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

    def _create_dataset(self):
        with tf.name_scope("dataset"):
            self.is_train = tf.placeholder(tf.int16)
            self.lr = tf.placeholder(tf.float32)
            self.dataset_record = tf.data.TFRecordDataset(self._train_path).prefetch(30000)
            self.parse_dataset = self.dataset_record.map(self.__parse_function).batch(self.batch_size)
            self.data_iterator = self.parse_dataset.make_initializable_iterator()
            self.dataset_test_record = tf.data.TFRecordDataset(self._test_path).prefetch(30000)
            self.parse_test_dataset = self.dataset_test_record.map(self.__parse_function).batch(self.batch_size)
            self.data_test_iterator = self.parse_test_dataset.make_initializable_iterator()
            (self._uid, self._history_vid, self._history_category,
             self._history_tag, self._history_kis, self._history_album,
             self._history_author, self._last_vid, self._last_category,
             self._last_tag, self._last_kis, self._last_album, self._last_author,
             self._expose_vid, self._expose_category, self._expose_tag,
             self._expose_kis, self._expose_album, self._expose_author, self._label_sample) = tf.cond(self.is_train > 0,
                                                                                                      lambda: self.data_iterator.get_next(),
                                                                                                      lambda: self.data_test_iterator.get_next())
            self._expose_vid_dense = tf.sparse_tensor_to_dense(self._expose_vid)
            self._expose_category_dense = tf.sparse_tensor_to_dense(self._expose_category)
            self._expose_tag_dense = tf.sparse_tensor_to_dense(self._expose_tag)
            self._expose_kis_dense = tf.sparse_tensor_to_dense(self._expose_kis)
            self._expose_album_dense = tf.sparse_tensor_to_dense(self._expose_album)
            self._expose_author_dense = tf.sparse_tensor_to_dense(self._expose_author)
            self._dense_label = tf.sparse_tensor_to_dense(self._label_sample)

    def _create_feed_data(self):
        with tf.name_scope("create_feed_data"):
            self.vv = tf.placeholder(tf.int32, shape=[None, None], name='vv')
            self.category = tf.placeholder(tf.int32, shape=[None, None], name='category')
            self.tag = tf.placeholder(tf.int32, shape=[None, None], name='tag')
            self.kis = tf.placeholder(tf.int32, shape=[None, None], name='kis')
            self.album = tf.placeholder(tf.int32, shape=[None, None], name='album')
            self.author = tf.placeholder(tf.int32, shape=[None, None], name='author')

            self.last_vv = tf.placeholder(tf.int32, shape=[None, None], name='last_vv')
            self.last_category = tf.placeholder(tf.int32, shape=[None, None], name='last_category')
            self.last_tag = tf.placeholder(tf.int32, shape=[None, None], name='last_tag')
            self.last_kis = tf.placeholder(tf.int32, shape=[None, None], name='last_kis')
            self.last_album = tf.placeholder(tf.int32, shape=[None, None], name='last_album')
            self.last_author = tf.placeholder(tf.int32, shape=[None, None], name='last_author')

    def _create_embedding(self):
        with tf.name_scope("embed"):
            self._item_matrix = tf.get_variable(shape=[self._item_size, self._item_esize], name="item_matrix")

            self._category_matrix = tf.get_variable(shape=[self._c_size, self._c_esize], name="category_matrix")

            self._tag_matrix = tf.get_variable(shape=[self._t_size, self._t_esize], name="tag_matrix")

            self._kis_matrix = tf.get_variable(shape=[self._k_size, self._k_esize], name="kis_matrix")

            self._album_matrix = tf.get_variable(shape=[self._a_size, self._a_esize], name="album_matrix")

            self._author_matrix = tf.get_variable(shape=[self._u_size, self._u_esize], name="author_matrix")
            self._bais = tf.get_variable(shape=[self._item_size], initializer=tf.zeros_initializer(), name="bais")

    def _create_input_context_feed(self):
        with tf.name_scope("input_context_feed"):
            self.vv_context = tf.nn.embedding_lookup(self._item_matrix, self.vv)
            self.category_context = tf.nn.embedding_lookup(self._category_matrix, self.category)
            self.tag_context = tf.nn.embedding_lookup(self._tag_matrix, self.tag)
            self.kis_context = tf.nn.embedding_lookup(self._kis_matrix, self.kis)
            self.album_context = tf.nn.embedding_lookup(self._album_matrix, self.album)
            self.author_context = tf.nn.embedding_lookup(self._author_matrix, self.author)

            self.last_vv_context = tf.nn.embedding_lookup(self._item_matrix, self.last_vv)
            self.last_category_context = tf.nn.embedding_lookup(self._category_matrix, self.last_category)
            self.last_tag_context = tf.nn.embedding_lookup(self._tag_matrix, self.last_tag)
            self.last_kis_context = tf.nn.embedding_lookup(self._kis_matrix, self.last_kis)
            self.last_album_context = tf.nn.embedding_lookup(self._album_matrix, self.last_album)
            self.last_author_context = tf.nn.embedding_lookup(self._author_matrix, self.last_author)

            self.hist_embed = tf.concat([self.vv_context, self.category_context, self.tag_context,
                                         self.kis_context, self.album_context, self.author_context], -1)
            self.last_embed = tf.concat([self.last_vv_context, self.last_category_context, self.last_tag_context,
                                         self.last_kis_context, self.last_album_context, self.last_author_context], -1)

            self.history_embed = tf.reduce_mean(self.hist_embed, 1)
            self.last_embed = tf.reduce_mean(self.last_embed, 1)
            self.input_context = tf.concat([self.history_embed, self.last_embed], -1)

    def _create_input_context_record(self):
        with tf.name_scope("input_context_record"):
            history_context = tf.nn.embedding_lookup_sparse(self._item_matrix,
                                                            self._history_vid,
                                                            combiner="mean",
                                                            sp_weights=None,
                                                            name="history_context")

            # 类别输入矩阵
            category_context = tf.nn.embedding_lookup_sparse(self._category_matrix,
                                                             self._history_category,
                                                             combiner="mean",
                                                             sp_weights=None,
                                                             name="category_context")

            # 标签输入矩阵
            tag_contex = tf.nn.embedding_lookup_sparse(self._tag_matrix,
                                                       self._history_tag,
                                                       combiner="mean",
                                                       sp_weights=None,
                                                       name="sc_context")

            # kis输入矩阵
            kis_context = tf.nn.embedding_lookup_sparse(self._kis_matrix,
                                                        self._history_kis,
                                                        combiner="mean",
                                                        sp_weights=None,
                                                        name="tc_context")

            # 专辑输入矩阵
            album_context = tf.nn.embedding_lookup_sparse(self._album_matrix,
                                                          self._history_album,
                                                          combiner="mean",
                                                          sp_weights=None,
                                                          name="album_context")

            author_context = tf.nn.embedding_lookup_sparse(self._author_matrix,
                                                           self._history_author,
                                                           combiner="mean",
                                                           sp_weights=None,
                                                           name="author_context")

            last_vid_context = tf.nn.embedding_lookup_sparse(self._item_matrix,
                                                             self._last_vid,
                                                             combiner="mean",
                                                             sp_weights=None,
                                                             name="last_vid_context")

            last_category_context = tf.nn.embedding_lookup_sparse(self._category_matrix,
                                                                  self._last_category,
                                                                  combiner="mean",
                                                                  sp_weights=None,
                                                                  name="last_category_context")

            last_tag_contex = tf.nn.embedding_lookup_sparse(self._tag_matrix,
                                                            self._last_tag,
                                                            combiner="mean",
                                                            sp_weights=None,
                                                            name="last_tag_context")

            last_kis_context = tf.nn.embedding_lookup_sparse(self._kis_matrix,
                                                             self._last_kis,
                                                             combiner="mean",
                                                             sp_weights=None,
                                                             name="last_kis_context")

            last_album_context = tf.nn.embedding_lookup_sparse(self._album_matrix,
                                                               self._last_album,
                                                               combiner="mean",
                                                               sp_weights=None,
                                                               name="last_album_context")

            last_author_context = tf.nn.embedding_lookup_sparse(self._author_matrix,
                                                                self._last_author,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="last_author_context")

            self.input_context = tf.concat([history_context,
                                            category_context,
                                            tag_contex,
                                            # kis_context,
                                            # album_context,
                                            author_context,
                                            last_vid_context,
                                            last_category_context,
                                            last_tag_contex,
                                            # last_kis_context,
                                            # last_album_context,
                                            last_author_context], -1)

    def _create_dnn_process(self):
        with tf.name_scope("dnn_process"):
            self.batch_normal = tf.layers.batch_normalization(inputs=self.input_context, name="batch_normal")
            layers1 = tf.layers.dense(self.batch_normal, 1024, activation=tf.nn.relu, name="layer1")
            layers1_dropout = tf.nn.dropout(layers1, keep_prob=self._keep_prob, name="dropout1")

            # 全连接层2
            layers2 = tf.layers.dense(layers1_dropout, 512, activation=tf.nn.relu, name="layer2")
            layers2_dropout = tf.nn.dropout(layers2, keep_prob=self._keep_prob, name="dropout2")

            # 全连接层3
            layers3 = tf.layers.dense(layers2_dropout,
                                      self._item_esize + self._c_esize + self._t_esize + self._k_esize + self._a_esize + self._u_esize,
                                      activation=tf.nn.relu,
                                      name="layer3")

            self._user_vector = tf.expand_dims(layers3, 1)

    def _create_smaple_conext(self):
        with tf.name_scope("sample_context"):
            self._expose_vid_context = tf.nn.embedding_lookup(self._item_matrix, self._expose_vid_dense)
            self._expose_category_context = tf.nn.embedding_lookup(self._category_matrix, self._expose_category_dense)
            self._expose_tag_context = tf.nn.embedding_lookup(self._tag_matrix, self._expose_tag_dense)
            self._expose_kis_context = tf.nn.embedding_lookup(self._kis_matrix, self._expose_kis_dense)
            self._expose_album_context = tf.nn.embedding_lookup(self._album_matrix, self._expose_album_dense)
            self._expose_author_context = tf.nn.embedding_lookup(self._author_matrix, self._expose_author_dense)
            self._sample_context = tf.concat([self._expose_vid_context,
                                              self._expose_category_context,
                                              self._expose_tag_context,
                                              # self._expose_kis_context,
                                              # self._expose_album_context,
                                              self._expose_author_context], -1)
            self._input_bais = tf.nn.embedding_lookup(self._bais, self._expose_vid_dense)

    def _create_sample_loss(self):
        with tf.name_scope("sample_loss"):
            self._distance = tf.squeeze(
                tf.matmul(tf.transpose(self._user_vector, perm=[0, 1, 2]),
                          tf.transpose(self._sample_context, perm=[0, 2, 1]), name="distance"),
                axis=1) + self._input_bais

            self._softmax_dist = tf.nn.softmax_cross_entropy_with_logits(labels=self._dense_label,
                                                                         logits=self._distance)

            self._sample_loss = tf.reduce_mean(self._softmax_dist)
            tf.cond(self.is_train > 0, lambda: tf.summary.scalar('train_loss', self._sample_loss),
                    lambda: tf.summary.scalar('test_loss', self._sample_loss))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self._optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self._sample_loss,
                                                                                  global_step=self.global_step)

    def _carete_candidate_embedding(self):
        with tf.name_scope("candidate"):
            self._vid_dense = tf.placeholder(tf.int32, shape=[None], name='vid_dense')
            self._category_dense = tf.placeholder(tf.int32, shape=[None], name='category_dense')
            self._tag_dense = tf.placeholder(tf.int32, shape=[None], name='tag_dense')
            self._kis_dense = tf.placeholder(tf.int32, shape=[None], name='kis_dense')
            self._album_dense = tf.placeholder(tf.int32, shape=[None], name='album_dense')
            self._author_dense = tf.placeholder(tf.int32, shape=[None], name='author_dense')

            self._all_vid_context = tf.nn.embedding_lookup(self._item_matrix, self._vid_dense)
            self._all_category_context = tf.nn.embedding_lookup(self._category_matrix, self._category_dense)
            self._all_tag_context = tf.nn.embedding_lookup(self._tag_matrix, self._tag_dense)
            self._all_kis_context = tf.nn.embedding_lookup(self._kis_matrix, self._kis_dense)
            self._all_album_context = tf.nn.embedding_lookup(self._album_matrix, self._album_dense)
            self._all_author_context = tf.nn.embedding_lookup(self._author_matrix, self._author_dense)
            self._all_bais_context = tf.expand_dims(tf.nn.embedding_lookup(self._bais, self._vid_dense), 1)
            self._all_embedd_context = tf.nn.l2_normalize(tf.concat([self._all_vid_context,
                                                                     self._all_category_context,
                                                                     self._all_tag_context,
                                                                     # self._all_kis_context,
                                                                     # self._all_album_context,
                                                                     self._all_author_context,
                                                                     self._all_bais_context], -1), 1)

    def _create_top_k(self):
        with tf.name_scope("top_k"):
            self._dims_bais = tf.ones_like(self._user_vector[:, :, 0:1])
            self._user_bais_vector = tf.concat([self._user_vector, self._dims_bais], -1)
            self._user_vector_squeeze = tf.squeeze(self._user_bais_vector, 1)
            distance = tf.matmul(self._user_bais_vector, self._all_embedd_context, transpose_b=True, name="l2_distance")
            self._top_values, self._top_idxs = tf.nn.top_k(distance, k=80)

    def build_graph(self):
        self._create_embedding()
        if self.predict_online:
            self._create_feed_data()
            self._create_input_context_feed()
            self._create_dnn_process()
            self._carete_candidate_embedding()
            self._create_top_k()
        else:
            self._create_dataset()
            self._create_input_context_record()
            self._create_dnn_process()
            self._create_smaple_conext()
            self._create_sample_loss()
            self._create_optimizer()
            self._carete_candidate_embedding()
            self._create_top_k()

    def split(self, vid_site):
        vid = vid_site[0:len(vid_site) - 1]
        site = vid_site[len(vid_site) - 1]
        return vid + "#" + site

    def parse_candidate(self):
        self._index_video = dict()
        self._vids = []
        self._categorys = []
        self._tags = []
        self._kiss = []
        self._albums = []
        self._authors = []
        for i, line in enumerate(open(self._candidate_path, "r")):
            res = line.strip().split("\t")
            vid = res[0]
            index = int(res[1])
            category = int(res[2])
            tag = int(res[3])
            kis = int(res[4])
            album = int(res[5])
            author = int(res[6])
            self._vids.append(index)
            self._categorys.append(category)
            self._tags.append(tag)
            self._kiss.append(kis)
            self._albums.append(album)
            self._authors.append(author)
            self._index_video[i] = vid

    def check_point(self, sess, saver, epoch):
        if os.path.exists(self._model_path) is False:
            os.mkdir(self._model_path)
        check_point = os.path.join(self._model_path, "check_point/dl")
        saver.save(sess, check_point, epoch)
        logging.info("Check point: {}, epoch: {}".format(check_point, epoch))

    def predict(self, output):
        writer = open(output, "w")
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)))

        ckpt = tf.train.get_checkpoint_state(os.path.join(self._model_path, "check_point/"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logging.info("tf model init successfully")
            sess.run(model.data_iterator.initializer)
            feed_dict = {model._vid_dense: np.array(self._vids, dtype="int32"),
                         model._category_dense: np.array(self._categorys, dtype="int32"),
                         model._tag_dense: np.array(self._tags, dtype="int32"),
                         model._kis_dense: np.array(self._kiss, dtype="int32"),
                         model._album_dense: np.array(self._albums, dtype="int32"),
                         model._author_dense: np.array(self._authors, dtype="int32"),
                         model.is_train: 1}
            _ = sess.run(model._vid_dense, feed_dict=feed_dict)
            index = 0
            while True:
                try:
                    uids, top_values, top_idxs = sess.run([model._uid, model._top_values, model._top_idxs],
                                                          feed_dict=feed_dict)
                    for uid, value, idx in zip(uids.values, top_values, top_idxs):
                        videos = []
                        for v, i in zip(value, idx):
                            vid = self._index_video[i]
                            videos.append(self.split(vid))
                        writer.write(str(uid, encoding="utf8") + "\t" + ",".join(videos) + "\n")
                        index += 1
                        if index % 10000 == 0:
                            logging.info("predict index: {}".format(index))
                except tf.errors.OutOfRangeError:
                    logging.info("predict finish!")
                    break

    def generate_video_embedding(self, output):
        writer = open(output, "w")
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)))

        ckpt = tf.train.get_checkpoint_state(os.path.join(self._model_path, "check_point/"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logging.info("tf model init successfully")
            feed_dict = {model._vid_dense: np.array(self._vids, dtype="int32"),
                         model._category_dense: np.array(self._categorys, dtype="int32"),
                         model._tag_dense: np.array(self._tags, dtype="int32"),
                         model._kis_dense: np.array(self._kiss, dtype="int32"),
                         model._album_dense: np.array(self._albums, dtype="int32"),
                         model._author_dense: np.array(self._authors, dtype="int32")}
            all_embedding = sess.run(model._all_embedd_context, feed_dict=feed_dict)
            for index, vector in enumerate(all_embedding):
                vec_str = "\t".join([str(x) for x in list(vector)])
                vid = self._index_video[index]
                writer.write(vid + "\t" + vec_str + "\n")
            writer.close()

    def predict_user_vector(self):
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))

        ckpt = tf.train.get_checkpoint_state(os.path.join(self._model_path, "check_point/"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logging.info("tf model init successfully")
            builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(self._model_path, "save_model/"))
            inputs = {'vv': tf.saved_model.utils.build_tensor_info(model.vv),
                      'category': tf.saved_model.utils.build_tensor_info(model.category),
                      'tag': tf.saved_model.utils.build_tensor_info(model.tag),
                      'kis': tf.saved_model.utils.build_tensor_info(model.kis),
                      'album': tf.saved_model.utils.build_tensor_info(model.album),
                      'author': tf.saved_model.utils.build_tensor_info(model.author),

                      'last_vv': tf.saved_model.utils.build_tensor_info(model.last_vv),
                      'last_category': tf.saved_model.utils.build_tensor_info(model.last_category),
                      'last_tag': tf.saved_model.utils.build_tensor_info(model.last_tag),
                      'last_kis': tf.saved_model.utils.build_tensor_info(model.last_kis),
                      'last_album': tf.saved_model.utils.build_tensor_info(model.last_album),
                      'last_author': tf.saved_model.utils.build_tensor_info(model.last_author)
                      }
            outputs = {'output': tf.saved_model.utils.build_tensor_info(model._user_bais_vector)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                               'tensorflow/serving/predict')
            builder.add_meta_graph_and_variables(sess, ['serve'], {'serving_default': signature})
            builder.save()

    def predict_test(self):
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))

        ckpt = tf.train.get_checkpoint_state(os.path.join(self._model_path, "check_point/"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logging.info("tf model init successfully")
            inputs_vid = [[500453, 2237622]]
            inputs_category = [[406, 649]]
            inputs_tag = [[10988, 63741]]
            inputs_kis = [[7688, 7688]]
            inputs_album = [[3593, 2393]]
            inputs_author = [[59688, 59688]]

            last_vid = [121040]
            last_category = [238]
            last_tag = [151356]
            last_kis = [7688]
            last_album = [2393]
            last_author = [59688]
            feed_dict = {
                model._vid_dense: np.array(self._vids, dtype="int32"),
                model._category_dense: np.array(self._categorys, dtype="int32"),
                model._tag_dense: np.array(self._tags, dtype="int32"),
                model._kis_dense: np.array(self._kiss, dtype="int32"),
                model._album_dense: np.array(self._albums, dtype="int32"),
                model._author_dense: np.array(self._authors, dtype="int32"),
                model.vv: np.array(inputs_vid, dtype="int32"),
                model.category: np.array(inputs_category, dtype="int32"),
                model.tag: np.array(inputs_tag, dtype="int32"),
                model.kis: np.array(inputs_kis, dtype="int32"),
                model.album: np.array(inputs_album, dtype="int32"),
                model.author: np.array(inputs_author, dtype="int32"),
                model.last_vv: np.array([last_vid], dtype="int32"),
                model.last_category: np.array([last_category], dtype="int32"),
                model.last_tag: np.array([last_tag], dtype="int32"),
                model.last_kis: np.array([last_kis], dtype="int32"),
                model.last_album: np.array([last_album], dtype="int32"),
                model.last_author: np.array([last_author], dtype="int32")
            }

            history = sess.run(self._top_idxs, feed_dict=feed_dict)
            # _all_embedd_context
            #
            print("===================\n\n")
            print(np.shape(history))
            print(history)

    def test_category_map(self, sess, epoch, batch_cnt):
        feed_dict = {model._vid_dense: np.array(self._vids, dtype="int32"),
                     model._category_dense: np.array(self._categorys, dtype="int32"),
                     model._tag_dense: np.array(self._tags, dtype="int32"),
                     model._kis_dense: np.array(self._kiss, dtype="int32"),
                     model._album_dense: np.array(self._albums, dtype="int32"),
                     model._author_dense: np.array(self._authors, dtype="int32"),
                     model.is_train: 0}
        total_category_true = 0
        total_category_all = 0
        batch_cnt = 0
        sess.run(model.data_test_iterator.initializer)
        while True:
            try:
                uids, top_values, top_idxs, vids_list = sess.run(
                    [model._uid, model._top_values, model._top_idxs, model._expose_vid_dense], feed_dict=feed_dict)
                cnt = 0
                all_cnt = 0
                for uid, value, idx, vids in zip(uids.values, top_values, top_idxs, vids_list):
                    index = list(vids)[-1]
                    source_category = self._categorys[index]
                    for value, vid_index in zip(value, idx):
                        target_category = self._categorys[vid_index]
                        if target_category == source_category:
                            cnt += 1
                        all_cnt += 1
                batch_cnt += 1
                total_category_true += cnt
                total_category_all += all_cnt
                if batch_cnt % 100 == 0:
                    logging.info(
                        "epoch :{}, test batch count:{}, batch acc ratio :{}".format(epoch, batch_cnt, cnt / all_cnt))
            except tf.errors.OutOfRangeError:
                logging.info("epoch :{}, batch count:{}, test average acc ratio :{}".format(epoch, batch_cnt,
                                                                                            total_category_true / total_category_all))

                break

    def train(self, epoch, lr):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            batch_cnt = 0
            total_loss = 0.0
            sess.run(self.data_iterator.initializer)

            while True:
                try:
                    feed = {model.is_train: 1, model.lr: lr}
                    loss, _, global_step = sess.run([model._sample_loss, model._optimizer, model.global_step],
                                                    feed_dict=feed)
                    batch_cnt += 1
                    total_loss += loss
                    if batch_cnt % 1000 == 0:
                        logging.info(
                            "epoch :{}, train batch count:{}, global step:{}, loss:{}".format(i, batch_cnt, global_step,
                                                                                              loss))
                except tf.errors.OutOfRangeError:
                    logging.info(
                        "epoch :{}, batch count:{}, train average loss:{}".format(i, batch_cnt, total_loss / batch_cnt))
                    self.check_point(sess, saver, i)
                    break

            self.test_category_map(sess, i, batch_cnt)

            feed = {model.is_train: 0, model.lr: lr}
            batch_cnt = 0
            total_loss = 0.0
            sess.run(model.data_test_iterator.initializer)
            while True:
                try:
                    self._keep_prob = 1.0
                    loss = sess.run(model._sample_loss, feed_dict=feed)
                    batch_cnt += 1
                    total_loss += loss
                    if batch_cnt % 1000 == 0:
                        logging.info("epoch :{}, test batch count:{}, loss:{}".format(i, model, loss))
                except tf.errors.OutOfRangeError:
                    logging.info(
                        "epoch :{}, batch count:{}, test average loss:{}".format(i, batch_cnt, total_loss / batch_cnt))
                    break


def build_input_file_queue(input_dir):
    files = os.listdir(input_dir)
    inputs = [os.path.join(input_dir, file) for file in files]
    return inputs


if __name__ == "__main__":
    #     ap = argparse.ArgumentParser()

    #     ap.add_argument("--INPUT", default="", type=str, help="path of input record")

    #     # 观影历史特征
    #     ap.add_argument("--VS", default=0, type=int, help="video size of input")
    #     ap.add_argument("--VES", default=128, type=int, help="video embedding size")

    #     # 一级类特征
    #     ap.add_argument("--FCS", default=0, type=int, help="first category size of input")
    #     ap.add_argument("--FCES", default=10, type=int, help="first category embedding size")

    #     # 二级类特征
    #     ap.add_argument("--SCS", default=0, type=int, help="second category size of input")
    #     ap.add_argument("--SCES", default=25, type=int, help="second category embedding size")

    #     # 三级类特征
    #     ap.add_argument("--TCS", default=0, type=int, help="third category size of input")
    #     ap.add_argument("--TCES", default=32, type=int, help="third category, embedding size")

    #     # 精编标签特征
    #     ap.add_argument("--LS", default=0, type=int, help="label size of input")
    #     ap.add_argument("--LES", default=32, type=int, help="label embedding size")

    #     # 普通标签特征
    #     ap.add_argument("--TS", default=0, type=int, help="tag size of input")
    #     ap.add_argument("--TES", default=128, type=int, help="tag embedding size")

    #     # 迭代次数
    #     ap.add_argument("--EPOCH", default=0, type=int, help="epoch of train model")
    #     ap.add_argument("--LR", default=0, type=float, help="learning rate of train model")

    #     # 模型路径
    #     ap.add_argument("--MODEL", default=None, type=str, help="path of model")

    #     args = vars(ap.parse_args())
    # files = build_input_file_queue(args["INPUT"])
    files = build_input_file_queue("/data/app/xuezhengyin/workspace/youtube/data_new/input_predict")
    train_files, test_files = train_test_split(files, test_size=0.01)
    model = BaseModel(files, test_files, "/data/app/xuezhengyin/workspace/youtube/data_new/candidate",
                      # 1909814, 128, #视频
                      # 696, 64,    #三级类
                      # 143827, 128,  #标签
                      # 18143, 0,   #kis
                      # 4908, 0,   #专辑
                      # 132792, 128, #作者
                      685483, 128,  # 视频
                      624, 64,  # 三级类
                      68807, 128,  # 标签
                      10744, 0,  # kis
                      4908, 0,  # 专辑
                      68433, 128,  # 作者
                      "/data/app/xuezhengyin/workspace/youtube/model_new", 0.9, True)
    model.parse_candidate()
    model.build_graph()
    # model.train(3, 1.0)
    # model.predict_test()
    model.predict_user_vector()
    # model.predict("/data/app/xuezhengyin/workspace/youtube/data_new/result")
    # model.generate_video_embedding("/data/app/xuezhengyin/workspace/youtube/data_new/video_vector")
