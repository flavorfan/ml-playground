#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 8:27
# @Author  : Flavorfan
# @File    : tf_autoencoder.py

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, GlobalMaxPooling2D

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

class TfAutoencoder():
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16, is_compiled=True):
        encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
        x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(3)(x)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(16, 3, activation='relu')(x)
        encoder_output = layers.GlobalMaxPooling2D()(x)

        encoder = keras.Model(encoder_input, encoder_output, name='encoder')
        encoder.summary()

        decoder_input = keras.Input(shape=(16,), name='encoded_img')
        x = layers.Reshape((4, 4, 1))(decoder_input)
        x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
        x = layers.UpSampling2D(3)(x)
        x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
        decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

        decoder = keras.Model(decoder_input, decoder_output, name='decoder')
        decoder.summary()

        autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
        encoded_img = encoder(autoencoder_input)
        decoded_img = decoder(encoded_img)
        autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
        autoencoder.summary()

        if is_compiled:
            opt = Adam(lr=1e-3)
            autoencoder.compile(loss="mse", optimizer=opt)

        return encoder, decoder, autoencoder

def TfEnsenmbleModel():
    def get_model():
        inputs = keras.Input(shape=(128,))
        outputs = layers.Dense(1)(inputs)
        return keras.Model(inputs, outputs)

    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = keras.Input(shape=(128,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
    return ensemble_model

class TfMultInputMultOutput():
    @staticmethod
    def build():
        num_tags = 12  # Number of unique issue tags
        num_words = 10000  # Size of vocabulary obtained when preprocessing text data
        num_departments = 4  # Number of departments for predictions

        title_input = keras.Input(shape=(None,), name='title')  # Variable-length sequence of ints
        body_input = keras.Input(shape=(None,), name='body')  # Variable-length sequence of ints
        tags_input = keras.Input(shape=(num_tags,), name='tags')  # Binary vectors of size `num_tags`

        # Embed each word in the title into a 64-dimensional vector
        title_features = layers.Embedding(num_words, 64)(title_input)
        # Embed each word in the text into a 64-dimensional vector
        body_features = layers.Embedding(num_words, 64)(body_input)

        # Reduce sequence of embedded words in the title into a single 128-dimensional vector
        title_features = layers.LSTM(128)(title_features)
        # Reduce sequence of embedded words in the body into a single 32-dimensional vector
        body_features = layers.LSTM(32)(body_features)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([title_features, body_features, tags_input])

        # Stick a logistic regression for priority prediction on top of the features
        priority_pred = layers.Dense(1, name='priority')(x)
        # Stick a department classifier on top of the features
        department_pred = layers.Dense(num_departments, name='department')(x)

        # Instantiate an end-to-end model predicting both priority and department
        model = keras.Model(inputs=[title_input, body_input, tags_input],
                            outputs=[priority_pred, department_pred])

        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=[keras.losses.BinaryCrossentropy(from_logits=True),
                            keras.losses.CategoricalCrossentropy(from_logits=True)],
                      loss_weights=[1., 0.2])

        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss={'priority': keras.losses.BinaryCrossentropy(from_logits=True),
                            'department': keras.losses.CategoricalCrossentropy(from_logits=True)},
                      loss_weights=[1., 0.2])

        # Dummy input data
        title_data = np.random.randint(num_words, size=(1280, 10))
        body_data = np.random.randint(num_words, size=(1280, 100))
        tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')

        # Dummy target data
        priority_targets = np.random.random(size=(1280, 1))
        dept_targets = np.random.randint(2, size=(1280, num_departments))

        model.fit({'title': title_data, 'body': body_data, 'tags': tags_data},
                  {'priority': priority_targets, 'department': dept_targets},
                  epochs=2,
                  batch_size=32)


class ToyResNet():
    @staticmethod
    def build():
        inputs = keras.Input(shape=(32, 32, 3), name='img')
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        block_1_output = layers.MaxPooling2D(3)(x)

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10)(x)

        model = keras.Model(inputs, outputs, name='toy_resnet')
        model.summary()

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['acc'])

        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=1,
                  validation_split=0.2)


def TfSharedLayers():
    # Embedding for 1000 unique words mapped to 128-dimensional vectors
    shared_embedding = layers.Embedding(1000, 128)
    # Variable-length sequence of integers
    text_input_a = keras.Input(shape=(None,), dtype='int32')
    # Variable-length sequence of integers
    text_input_b = keras.Input(shape=(None,), dtype='int32')
    # Reuse the same layer to encode both inputs
    encoded_input_a = shared_embedding(text_input_a)
    encoded_input_b = shared_embedding(text_input_b)

def ExtractReuseNodesInGraphLayers():
    vgg19 = tf.keras.applications.VGG19()
    features_list = [layer.output for layer in vgg19.layers]
    feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

    img = np.random.random((1, 224, 224, 3)).astype('float32')
    extracted_features = feat_extraction_model(img)

####################### auto encoder ################


class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z



class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_output = layers.Dense(original_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed



class TfVae():
    @staticmethod
    def build():
        original_dim = 784
        intermediate_dim = 64
        latent_dim = 32

        # Define encoder model.
        original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
        x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()((z_mean, z_log_var))
        encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

        # Define decoder model.
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(original_dim, activation='sigmoid')(x)
        decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

        # Define VAE model.
        outputs = decoder(z)
        vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        vae.add_loss(kl_loss)

        # Train.
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        # vae.fit(x_train, x_train, epochs=3, batch_size=64)

        return vae

def train_vae(x_train):
    vae = VariationalAutoEncoder(784, 64, 32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(x_train, x_train, epochs=3, batch_size=64)