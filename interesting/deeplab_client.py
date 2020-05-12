#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/12 11:39
# @Author  : Flavorfan
# @File    : deeplab_client.py

from __future__ import print_function
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
# from StringIO import StringIO
from io import StringIO
from io import BytesIO
# try:
#     from StringIO import StringIO ## for Python 2
# except ImportError:
#     from io import StringIO ## for Python 3


if __name__ == '__main__':
    server = '192.168.62.70:9000'
    host, port = server.split(':')

    # image_url = "https://www.publicdomainpictures.net/pictures/60000/nahled/bird-1382034603Euc.jpg"

    image_url = "http://5b0988e595225.cdn.sohucs.com/images/20180113/b9ef45d6babf492abd1cd79ce18eef8b.jpeg"

    response = requests.get(image_url)
    image = np.array(Image.open(BytesIO(response.content)))
    height = image.shape[0]
    width = image.shape[1]
    print("Image shape:", image.shape)

    response = requests.get(image_url)
    image = np.array(Image.open(StringIO(response.content)))
    height = image.shape[0]
    width = image.shape[1]
    print("Image shape:", image.shape)

