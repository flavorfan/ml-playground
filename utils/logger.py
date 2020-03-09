#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 14:15
# @Author  : Flavorfan
# @File    : logger.py

import logging


def root_logger(file_path: object = None) -> object:
    logFormatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    # fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    if file_path:
        fileHandler = logging.FileHandler(file_path)
    else:
        fileHandler = logging.FileHandler('./log/log.txt')
    fileHandler.setFormatter(logFormatter)

    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)