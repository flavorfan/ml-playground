#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 16:19
# @Author  : Flavorfan
# @File    : load_test_autoencoder.py

import matplotlib
matplotlib.use("Agg")

import os

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse

# import cv2

from autoencoder.fan_autoencoder import FanAutoencoder
from autoencoder.fan_cnn_autoencoder import FanCnnAutoencoder
from visualization.plot_autoencoder_result import training_plot

# log
from utils import root_logger
import logging

def arg_parse():
    # global args
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # select model type : fc, cnn,
    ap.add_argument("-t", "--model_type", type=str, default="fc",
                    help="# select model_type from fc,rnn,and so on  ")

    ap.add_argument("-s", "--samples", type=int, default=8,
                    help="# number of samples to visualize when decoding")
    ap.add_argument("-d", "--n_dims", nargs='+', type=int,
                    help="n_dim of layers")
    ap.add_argument("-c", "--code_dim", type=int, default=16,
                    help="# code_dim - latent layer size ")
    # train param
    ap.add_argument("-e", "--epochs", type=int, default=25,
                    help="# epochs  ")
    ap.add_argument("-b", "--batch_size", type=int, default=128,
                    help="# epochs  ")

    # model instance name
    ap.add_argument("-m", "--model_name", type=str, default="fan_fc_ae",
                    help="# model instance name, also for save the file  ")

    args = vars(ap.parse_args())

    if 'n_dims' not in args:
        args['n_dims'] = [256, 128]

    return args


def load_mnist_data():
    # load the MNIST dataset
    logging.info(" loading MNIST dataset...")
    ((trainX, _), (testX, _)) = mnist.load_data()
    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    trainX = np.expand_dims(trainX, axis=-1)  # (sample,w,h) -> (sample, w, h, d)
    testX = np.expand_dims(testX, axis=-1)
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    return trainX, testX


if __name__ == '__main__':
    # init
    args = arg_parse()
    root_logger('./logs/log.txt')
    logging.info(str(args))

    # load model
    ckpts_path = 'ckpts/' + args['model_name']

    # (encoder, decoder, autoencoder) = FanAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])
    if args['model_type'] == 'rnn':
        (encoder, decoder, autoencoder) = FanCnnAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])
    elif args['model_type'] == 'fc':
        (encoder, decoder, autoencoder) = FanAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])
    else:  #
        (encoder, decoder, autoencoder) = FanAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])

    opt = Adam(lr=1e-3)
    autoencoder.compile(loss="mse", optimizer=opt)
    latest = tf.train.latest_checkpoint(ckpts_path)
    autoencoder.load_weights(latest)

    # data
    trainX, testX = load_mnist_data()

    code = encoder.predict(testX)
    logging.info(str(code[0]))
    decoded = decoder.predict(code)

    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    output_name = 'output/{}_{}.png'.format(args['model_name'],time_str)
    plt.figure(figsize=(20, 4))
    n = args["samples"]
    for i in range(n):
        original = (testX[i].reshape(28, 28) * 255).astype("uint8")
        recon = (decoded[i].reshape(28, 28) * 255).astype("uint8")

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # if savename:
        #     plt.savefig(savename)
        plt.savefig(output_name)

    logging.info(str(code))

    logging.info(" Done!")