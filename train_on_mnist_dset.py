#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 9:50
# @Author  : Flavorfan
# @File    : train_on_mnist_dset.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
from utils import root_logger
import logging

from autoencoder.fan_autoencoder import FanVariationalAutoEncoder
from visualization import training_plot

def load_mnist_dataset():
    ((trainX, _), (testX, _)) = tf.keras.datasets.mnist.load_data()

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    trainX = np.expand_dims(trainX, axis=-1)  # (sample,w,h) -> (sample, w, h, d)
    testX = np.expand_dims(testX, axis=-1)

    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    train_data = tf.data.Dataset.from_tensor_slices((trainX, trainX))
    train_data = train_data.shuffle(5000).batch(64)

    test_data = tf.data.Dataset.from_tensor_slices((testX, testX))
    test_data = test_data.shuffle(5000).batch(64)

    return train_data, test_data

def arg_parse():
    # global args
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    # select model type : fc, cnn,
    ap.add_argument("-t", "--model_type", type=str, default="fc",
                    help="# select model_type from fc,rnn,vae and so on  ")

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
    ap.add_argument("-m", "--model_name", type=str, default="fan_vac",
                    help="# model instance name, also for save the file  ")

    args = vars(ap.parse_args())

    if 'n_dims' not in args:
        args['n_dims'] = [256, 128]

    return args


def make_cb_dirs(model_name):
    # create the folder for ckpts and tb logs
    ckpts_path = 'ckpts/' + model_name
    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)
    # checkpoint_dir = os.path.join(os.getcwd(), ckpts_path)
    checkpoint_path = ckpts_path + "/model_{epoch:04d}.ckpt"
    tb_log_path = 'logs/' + model_name
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path)

    return checkpoint_path, tb_log_path


def training_plot_and_savemodel():
    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    plot_name = 'training_plot/{}_{}.png'.format(args['model_name'], time_str)
    training_plot(H, plot_name, EPOCHS)

    path_to_save_model = 'output/save_models/{}_{}'.format(args['model_name'], time_str)
    autoencoder.save(path_to_save_model, 'tf')


if __name__ == '__main__':
    args = arg_parse()
    root_logger('./logs/log.txt')
    logging.info(str(args))

    checkpoint_path, tb_log_path = make_cb_dirs(args['model_name'])
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            period=5,
            verbose=1),
        TensorBoard(log_dir=tb_log_path)
    ]


    train_data, test_data = load_mnist_dataset()
    encoder, decoder, autoencoder = FanVariationalAutoEncoder.build(is_compiled=True)

    EPOCHS = args['epochs']  # 25
    BS = args['batch_size']  # 32

    H = autoencoder.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS)

    training_plot_and_savemodel()



    # output_name = 'output/{}_{}.png'.format(args['model_name'], time_str)
    # plt.figure(figsize=(20, 4))
    # n = args["samples"]
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     original = (testX[i].reshape(28, 28) * 255).astype("uint8")
    #     recon = (decoded[i].reshape(28, 28) * 255).astype("uint8")
    #
    #     plt.imshow(original)
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(recon)
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     # if savename:
    #     #     plt.savefig(savename)
    #     plt.savefig(output_name)