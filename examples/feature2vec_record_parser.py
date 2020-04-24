#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from deeprec.dataset.parser import BaseRecordParser, ParserType


def _parse_weight_mean(line) -> str:
    res = line.strip().split(" ")
    uid = bytes(res[0], encoding="utf-8")
    input_data = res[1].split(";")
    vid = int(input_data[0])
    third_category = int(input_data[1])
    tag = int(input_data[2].split(","))
    kis = int(input_data[3])
    album = int(input_data[4])
    output_data = res[2].split(";")
    output = int(output_data[0])

    feature = {
        "uid": BaseRecordParser.bytes_list_feature([uid]),
        "vid": BaseRecordParser.int64_list_feature([vid]),
        "third_category": BaseRecordParser.int64_list_feature(([third_category])),
        "tag": BaseRecordParser.int64_list_feature([tag]),
        "kis": BaseRecordParser.int64_list_feature([kis]),
        "album": BaseRecordParser.int64_list_feature([album]),
        "output": BaseRecordParser.int64_list_feature([output])
    }

    serial_string = tf.train.Example(features=tf.train.Features(feature=feature))
    return serial_string.SerializeToString()


def _parse_concat(line) -> str:
    res = line.strip().split(" ")
    uid = bytes(res[0], encoding="utf-8")

    item_features = [int(x) for x in res[1].split(";")]
    feature = {
        "uid": Feature2vecRecordParser.bytes_list_feature([uid]),
        "features": Feature2vecRecordParser.int64_list_feature(item_features)
    }

    serial_string = tf.train.Example(features=tf.train.Features(feature=feature))
    return serial_string.SerializeToString()


class Feature2vecRecordParser(BaseRecordParser):

    def __init__(self, path,
                 output,
                 combine="concat",
                 prefix="feature2vec",
                 suffix="record",
                 size=50000,
                 process=ParserType.TRAIN,
                 verbose=True):
        self._combine = combine
        super(Feature2vecRecordParser, self).__init__(path, output, prefix, suffix, size, process, verbose)

    def parse_train(self, line) -> str:
        """

        :param line: input data record line
        :return: serial string of record
        """
        if self._combine == "concat":
            return _parse_concat(line)

        if self._combine == "combine":
            return _parse_weight_mean(line)

    def parse_predict(self, line):
        pass

    def parse_test(self, line):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, type=str, help="input path of train dataset", required=True)
    ap.add_argument("--output", default=None, type=str, help="output path of record", required=True)
    ap.add_argument("--process", default=None, type=str, help="process of parser:[TRAIN, PREDICT]", required=True)
    ap.add_argument("--size", default=50000, type=int, help="size of record file")
    ap.add_argument("--combine", default=None, type=str, help="method of combine feature vector", required=True)

    args = vars(ap.parse_args())
    if args["process"] == "TRAIN":
        parser = Feature2vecRecordParser(path=args["input"],
                                         output=args["output"], process=ParserType.TRAIN)
        parser.dump()
    if args["process"] == "PREDICT":
        parser = Feature2vecRecordParser(path=args["input"],
                                         output=args["output"], process=ParserType.PREDICT)
        parser.dump()
