#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
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

    def _parser_train(self, line):
        pass

    def _parser_predict(self, line):
        pass

    def _parser_test(self, line):
        pass

    def _serial_string(self):
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