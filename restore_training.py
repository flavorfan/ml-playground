#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 10:12
# @Author  : Flavorfan
# @File    : restore_training.py

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
from visualization.plot_autoencoder_result import training_plot

# log
from utils import root_logger
import logging

def arg_parse():
    # global args
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
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


# def train_and_checkpoint(net, manager):
#     ckpt.restore(manager.latest_checkpoint)
#     if manager.latest_checkpoint:
#         logging.info("Restored from {}".format(manager.latest_checkpoint))
#     else:
#         logging.info("Initializing from scratch.")


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
    # data
    trainX, testX = load_mnist_data()

    # checkpoints save and restore
    ckpts_path = 'ckpts/' + args['model_name']
    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)

    tb_log_path = 'logs/' + args['model_name']
    if not os.path.exists(tb_log_path):
        os.makedirs(tb_log_path)



    opt = Adam(lr=1e-3)
    (encoder, decoder, autoencoder) = FanAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])
    autoencoder.compile(loss="mse", optimizer=opt)

    # load the latest ckpt
    save_dir = os.path.join(os.getcwd(), ckpts_path)
    # latest = tf.train.latest_checkpoint(save_dir)
    logging.info(save_dir)
    # logging.info(latest)
    latest = '/home/algo/code/gitrepo/pylib/ml-playground/ckpts/fc_1000_500_250_2/model_94.hdf5'
    autoencoder.load_weights(latest)

    # loss,acc = autoencoder.evaluate(testX, testX) # may be wrong
    # logging.info("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    callbacks = [
        ModelCheckpoint(
            filepath=ckpts_path + '/fc_ae_{epoch}',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        TensorBoard(log_dir=tb_log_path)
    ]

    EPOCHS = args['epochs']  # 25
    BS = args['batch_size']  # 32

    H = autoencoder.fit(
        trainX, trainX,
        validation_data=(testX, testX),
        epochs=EPOCHS,
        batch_size=BS,
        callbacks=callbacks)

    now = datetime.datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    plot_name = 'training_plot/{}_{}.png'.format(args['model_name'],time_str)
    training_plot(H, plot_name, EPOCHS)

    logging.info(" making predictions...")
    decoded = autoencoder.predict(testX)
    outputs = None

    output_name = 'output/{}_{}.png'.format(args['model_name'],time_str)
    plt.figure(figsize=(20, 4))
    n = args["samples"]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        original = (testX[i].reshape(28, 28) * 255).astype("uint8")
        recon = (decoded[i].reshape(28, 28) * 255).astype("uint8")

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

    logging.info(" Done!")

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=autoencoder)
    # manager = tf.train.CheckpointManager(ckpt, ckpts_path, max_to_keep=3)

    # train_and_checkpoint(autoencoder, manager)

