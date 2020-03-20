#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/18 9:50
# @Author  : Flavorfan
# @File    : train_on_mnist_dset.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


import numpy as np
from    PIL import Image

import matplotlib.pyplot as plt
import argparse
import os
import datetime
from utils import root_logger
import logging


from autoencoder.fan_autoencoder import FanVariationalAutoEncoder
from visualization import training_plot, grid_img

def load_mnist_dataset(batch_size = 64):
    (x_train, y_train_), (x_test, y_test_) = tf.keras.datasets.mnist.load_data()

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    # trainX = np.expand_dims(trainX, axis=-1)  # (sample,w,h) -> (sample, w, h, d)
    # testX = np.expand_dims(testX, axis=-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    train_data = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    train_data = train_data.shuffle(5000).batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    test_data = test_data.shuffle(5000).batch(batch_size)

    return train_data, test_data

def arg_parse():
    # global args
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    # select model type : fc, cnn,
    ap.add_argument("-t", "--model_type", type=str, default="vae",
                    help="# select model_type from fc,rnn,vae and so on  ")

    ap.add_argument("-s", "--samples", type=int, default=8,
                    help="# number of samples to visualize when decoding")
    #
    # ap.add_argument("-d", "--n_dims", nargs='+', type=int,
    #                 help="n_dim of layers")
    # ap.add_argument("-c", "--code_dim", type=int, default=16,
    #                 help="# code_dim - latent layer size ")

    # train param
    ap.add_argument("-e", "--epochs", type=int, default=100,
                    help="# epochs  ")
    ap.add_argument("-b", "--batch_size", type=int, default=100,
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


def train():
    global EPOCHS, H
    EPOCHS = args['epochs']  # 25
    checkpoint_path, tb_log_path = make_cb_dirs(args['model_name'])
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            period=5,
            verbose=1),
        TensorBoard(log_dir=tb_log_path)
    ]
    H = autoencoder.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS)
    training_plot_and_savemodel()


# 10 x 10 grid img
# def grid_img(x_concat, filename):
#     index = 0
#     new_im = Image.new('L', (280, 280))
#     for i in range(0, 280, 28):
#         for j in range(0, 280, 28):
#             im = x_concat[index]
#             im = Image.fromarray(im, mode='L')
#             new_im.paste(im, (i, j))
#             index += 1
#     new_im.save(filename)
#     plt.imshow(np.asarray(new_im))
#     plt.show()


if __name__ == '__main__':
    args = arg_parse()
    root_logger('./logs/log.txt')
    logging.info(str(args))

    batch_size = args['batch_size']
    epochs = args['epochs']
    n = args['samples']

    #
    train_data, test_data = load_mnist_dataset(batch_size)
    encoder, decoder, autoencoder = FanVariationalAutoEncoder.build(is_compiled=False)

    # train()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics.
    # train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    # val_acc_metric   = tf.keras.metrics.SparseCategoricalAccuracy()

    autoencoder.summary()
    # tf.keras.utils.plot_model(autoencoder, 'fan_vae.png', show_shapes=True)
    # image grid


    num_batches = 60000 // batch_size
    # loss_history = []
    prt_one = 1
    for epoch in range(epochs):
        logging.info('start of epoch %d' %(epoch,))
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits, z_mean, z_log_var = autoencoder(x_batch_train, training = True)
                # compute the loss value for this minbatch
                reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch_train, logits = logits)
                if prt_one:
                    logging.info(repr(reconstruction_loss))
                reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size
                if prt_one:
                    logging.info(repr(reconstruction_loss))
                kl_loss = - 0.5 * tf.reduce_sum(1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis= -1)
                if prt_one:
                    logging.info(repr(kl_loss))
                kl_loss = tf.reduce_mean(kl_loss)
                if prt_one:
                    logging.info(repr(kl_loss))
                prt_one = 0
                loss_value = tf.reduce_mean(reconstruction_loss) + kl_loss
            # loss_history.append(loss_value.numpy().mean())
            grads = tape.gradient(loss_value, autoencoder.trainable_weights)
            # mark ?
            # for g in grads:
            #     tf.clip_by_norm(g, 15)

            optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))
            if (step + 1) % 50 == 0:
                # logging.info('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                # logging.info('Seen so far: %s sample' % ((step+1) * batch_size))
                logging.info("Epoch[{}/{}], Step [{}/{}], loss: {:.4f} Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, epochs, step + 1, num_batches, float(loss_value),float(reconstruction_loss), float(kl_loss)))

        out_logits, _, _ = autoencoder(x_batch_train[:batch_size // 2])
        out = tf.nn.sigmoid(out_logits)  # out is just the logits, use sigmoid
        out = tf.reshape(out, [-1, 28, 28]).numpy() * 255

        x = tf.reshape(x_batch_train[:batch_size // 2], [-1, 28, 28])

        x_concat = tf.concat([x, out], axis=0).numpy() * 255.
        x_concat = x_concat.astype(np.uint8)

        grid_img(x_concat,'output/images2/vae_reconstructed_epoch_%d.png' % (epoch + 1))

        logging.info('New images saved !')









    # plot_name = 'training_plot/{}_{}.png'.format(args['model_name'], time_str)
    # training_plot(loss_history, plot_name,epochs)

    # testX = test_data.take(1)

    # logging.info(" making predictions...")

    # outputs = None

    # loop over our number of output
    #
    # ((trainX, _), (testX, _)) = tf.keras.datasets.mnist.load_data()
    #
    # # add a channel dimension to every image in the dataset, then scale
    # # the pixel intensities to the range [0, 1]
    # trainX = np.expand_dims(trainX, axis=-1)  # (sample,w,h) -> (sample, w, h, d)
    # testX = np.expand_dims(testX, axis=-1)
    #
    # trainX = trainX.astype("float32") / 255.0
    # testX = testX.astype("float32") / 255.0
    #
    #
    #
    # output_name = 'output/{}_{}.png'.format(args['model_name'], time_str)
    # plt.figure(figsize=(20, 4))

    # decoded = autoencoder.predict(testX)
    # logging.info(str(type(decoded)))

    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #
    #     input_img = np.expand_dims(testX[i],axis=0)
    #
    #     decoded , _ , _ = autoencoder.predict(input_img)
    #     original = (testX[i].reshape(28, 28) * 255).astype("uint8")
    #     recon =  (decoded.reshape(28, 28) * 255).astype("uint8")
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

    # logging.info(" Done!")
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