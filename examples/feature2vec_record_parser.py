#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from deeprec.dataset.parser import BaseRecordParser, ParserType


class Feature2vecRecordParser(BaseRecordParser):

    def __init__(self, path,
                 output,
                 prefix="feature2vec",
                 suffix="record",
                 size=50000,
                 process=ParserType.TRAIN,
                 verbose=True):
        super(Feature2vecRecordParser, self).__init__(path, output, prefix, suffix, size, process, verbose)

    def parse_train(self, line):
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
            "uid": Feature2vecRecordParser.bytes_list_feature([uid]),
            "vid": Feature2vecRecordParser.int64_list_feature([vid]),
            "third_category": Feature2vecRecordParser.int64_list_feature(([third_category])),
            "tag": Feature2vecRecordParser.int64_list_feature([tag]),
            "kis": Feature2vecRecordParser.int64_list_feature([kis]),
            "album": Feature2vecRecordParser.int64_list_feature([album]),
            "output": Feature2vecRecordParser.int64_list_feature([output])
        }

        serial_string = tf.train.Example(features=tf.train.Features(feature=feature))
        return serial_string.SerializeToString()

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

    args = vars(ap.parse_args())
    if args["process"] == "TRAIN":
        parser = Feature2vecRecordParser(path=args["input"],
                                         output=args["output"], process=ParserType.TRAIN)
        parser.dump()
    if args["process"] == "PREDICT":
        parser = Feature2vecRecordParser(path=args["input"],
                                         output=args["output"], process=ParserType.PREDICT)
        parser.dump()
