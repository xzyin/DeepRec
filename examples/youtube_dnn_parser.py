#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from deeprec.dataset.parser import BaseRecordParser, ParserType


def _parse_last(line, predict=False) -> str:
    res = line.strip().split(" ")
    uid = bytes(res[0], encoding="utf-8")

    history = res[1].split(";")
    # input context of video and features

    history_vid = [int(x) for x in history[0].split(",")]
    history_category = [int(x) for x in history[1].split(",")]
    history_tag = [int(x) for x in history[2].split(",")]
    history_kis = [int(x) for x in history[3].split(",")]
    history_album = [int(x) for x in history[4].split(",")]
    history_author = [int(x) for x in history[5].split(",")]

    # input of last video view and features
    last_view = res[2].split(";")
    last_vid = int(last_view[0])
    last_category = int(last_view[1])
    last_tag = int(last_view[2])
    last_kis = int(last_view[3])
    last_album = int(last_view[4])
    last_author = int(last_view[5])

    # input of positive sample video and features
    if predict is False:
        label = res[3].split(";")
        label_vid = int(label[0])
        label_category = int(label[1])
        label_tag = int(label[2])
        label_kis = int(label[3])
        label_album = int(label[4])
        label_author = int(label[5])

        # input of negative sample video and features
        expose = res[4].split(";")
        expose_vid = [int(x) for x in expose[0].split(",")]
        expose_category = [int(x) for x in expose[1].split(",")]
        expose_tag = [int(x) for x in expose[2].split(",")]
        expose_kis = [int(x) for x in expose[3].split(",")]
        expose_album = [int(x) for x in expose[4].split(",")]
        expose_author = [int(x) for x in expose[5].split(",")]

        label_sample = [0.0 for _ in expose_vid]
        expose_vid.append(label_vid)
        expose_category.append(label_category)
        expose_tag.append(label_tag)
        expose_kis.append(label_kis)
        expose_album.append(label_album)
        expose_author.append(label_author)

        label_sample.append(1.0)
    else:
        expose_vid = None
        expose_category = None
        expose_tag = None
        expose_kis = None
        expose_album = None
        expose_author = None
        label_sample = None

    feature = {
        "uid": BaseRecordParser.bytes_list_feature([uid]),
        "history_vid": BaseRecordParser.int64_list_feature(history_vid),
        "history_category": BaseRecordParser.int64_list_feature(history_category),
        "history_tag": BaseRecordParser.int64_list_feature(history_tag),
        "history_kis": BaseRecordParser.int64_list_feature(history_kis),
        "history_album": BaseRecordParser.int64_list_feature(history_album),
        "history_author": BaseRecordParser.int64_list_feature(history_author),

        "last_vid": BaseRecordParser.int64_list_feature([last_vid]),
        "last_category": BaseRecordParser.int64_list_feature([last_category]),
        "last_tag": BaseRecordParser.int64_list_feature([last_tag]),
        "last_kis": BaseRecordParser.int64_list_feature([last_kis]),
        "last_album": BaseRecordParser.int64_list_feature([last_album]),
        "last_author": BaseRecordParser.int64_list_feature([last_author]),

        "expose_vid": BaseRecordParser.int64_list_feature(expose_vid),
        "expose_category": BaseRecordParser.int64_list_feature(expose_category),
        "expose_tag": BaseRecordParser.int64_list_feature(expose_tag),
        "expose_kis": BaseRecordParser.int64_list_feature(expose_kis),
        "expose_album": BaseRecordParser.int64_list_feature(expose_album),
        "expose_author": BaseRecordParser.int64_list_feature(expose_author),
        "label_sample": BaseRecordParser.float_list_feature(label_sample),
    }
    serial_string = tf.train.Example(features=tf.train.Features(feature=feature))

    return serial_string.SerializeToString()



class YouTubeDNNRecordParser(BaseRecordParser):

    def __init__(self, path,
                 output,
                 input_format="last",
                 prefix="feature2vec",
                 suffix="record",
                 size=50000,
                 process=ParserType.TRAIN,
                 verbose=True):
        self._format = input_format
        super(YouTubeDNNRecordParser, self).__init__(path, output, prefix, suffix, size, process, verbose)

    def parse_train(self, line) -> str:
        """

        :param line: input data record line
        :return: serial string of record
        """
        if self._format == "last":
            return _parse_last(line)

    def parse_predict(self, line):
        if self._format == "last":
            return _parse_last(line, True)

    def parse_test(self, line):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, type=str, help="input path of train dataset", required=True)
    ap.add_argument("--output", default=None, type=str, help="output path of record", required=True)
    ap.add_argument("--process", default=None, type=str, help="process of parser:[TRAIN, PREDICT]", required=True)
    ap.add_argument("--size", default=50000, type=int, help="size of record file")
    ap.add_argument("--format", default=None, type=str, help="method of combine feature vector", required=True)

    args = vars(ap.parse_args())
    if args["process"] == "TRAIN":
        parser = YouTubeDNNRecordParser(path=args["input"],
                                        output=args["output"], process=ParserType.TRAIN)
        parser.dump()
    if args["process"] == "PREDICT":
        parser = YouTubeDNNRecordParser(path=args["input"],
                                        output=args["output"], process=ParserType.PREDICT)
        parser.dump()