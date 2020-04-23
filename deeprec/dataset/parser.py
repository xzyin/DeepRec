#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import abc
import logging
from enum import Enum
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ParserType(Enum):
    TRAIN = 1
    PREDICT = 2
    TEST = 3


class BaseRecordParser(metaclass=abc.ABCMeta):
    """

    """

    def __init__(self, path,
                 output,
                 prefix="dataset",
                 suffix="record",
                 size=-1,
                 process=ParserType.TRAIN,
                 verbose=True):

        self._input = path
        self._output = output
        self._size = size
        self._prefix = prefix
        self._suffix = suffix
        self._process = process
        self._verbose = verbose

    def dump(self):
        if self._size > 0:
            files_index = 0
            file_size = 0
            file_path = os.path.join("{}/{}.{}.{}".format(self._output, self._prefix, files_index, self._suffix))
            writer = tf.io.TFRecordWriter(file_path)
            for index, line in enumerate(open(self._input, "r")):
                serial = self.parser(line)
                writer.write(serial)
                file_size += 1
                if file_size >= self._size:
                    files_index += 1
                    file_size = 0
                    writer.close()
                    file_path = os.path.join(
                        "{}/{}.{}.{}".format(self._output, self._prefix, files_index, self._suffix))
                    writer = tf.io.TFRecordWriter(file_path)
                if index % 10000 == 0 and self._verbose:
                    logging.info("build {}.{} index:{}".format(self._prefix, self._suffix, index))
            writer.close()
        else:
            writer = tf.io.TFRecordWriter(os.path.join("{}/{}.{}".format(self._output, self._prefix, self._suffix)))
            for index, line in enumerate(open(self._input, "r")):
                serial = self.parser(line)
                writer.write(serial)
                if index % 10000 == 0 and self._verbose:
                    logging.info("build {}.{} index:{}".format(self._prefix, self._suffix, index))
            writer.close()
        if self._verbose:
            logging.info("build {}.{} successful! output path:{}".format(self._prefix, self._suffix, self._output))

    def parser(self, line) -> str:
        """
        :param line: input record line
        :return:
        """
        assert self._process not in ParserType.__members__

        if self._process == ParserType.TRAIN:
            return self.parse_train(line)

        if self._process == ParserType.PREDICT:
            return self.parse_predict(line)

        if self._process == ParserType.TEST:
            return self.parse_test(line)

    @abc.abstractmethod
    def parse_train(self, line) -> str:
        pass

    @abc.abstractmethod
    def parse_predict(self, line) -> str:
        pass

    @abc.abstractmethod
    def parse_test(self, line) -> str:
        pass

    @staticmethod
    def bytes_list_feature(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

    @staticmethod
    def int64_list_feature(x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

    @staticmethod
    def float_list_feature(x):
        return tf.train.Feature(float_list=tf.train.FloatList(value=x))

    @staticmethod
    def features_list_feature(x):
        return tf.train.FeatureList(feature=x)
