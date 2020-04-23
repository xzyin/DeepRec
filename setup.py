#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="DeepRec",
    version="0.1.1",
    author="xzyin",
    package=["deeprec"],
    url="https://github.com/xzyin/DeepRec",
    author_email="xuezhengyin@gmail.com",
    extras_require={
        "cpu": ["tensorflow==1.14.0"],
        "gpu": ["tensorflow-gpu==1.14.0"],
    },
    long_description="DeepRec"
)