
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model

class FanAutoencoder:
    @staticmethod
    def build(width,height,depth, n_dims=[256,128], code_dim=16):
        input_shape = (width, height, depth)

        # define the input to the encoder
        inputs = Input(shape= input_shape)
        x = Flatten()(inputs)

        for dim in n_dims:
            x = Dense(dim, activation=tf.nn.relu)(x)
        latent = Dense(code_dim, activation=tf.nn.relu)(x)
        

        #build the encoder model
        encoder = Model(inputs, latent, name="encoder")
        print(encoder.summary())

        # build the decoder
        latent_inputs = Input(shape=(code_dim,))
        x = Dense(n_dims[-1],activation=tf.nn.relu )(latent_inputs)
        # assert len

        # when layer >= 2
        if len(n_dims) >1:
            for dim in n_dims[-2::-1]:
                x = Dense(dim, activation=tf.nn.relu)(x)

        x = Dense(width * height * depth, activation=tf.nn.sigmoid)(x)

        outputs = Reshape((width, height, depth))(x)
        decoder = Model(latent_inputs, outputs, name="decoder")
        print(decoder.summary())

        autoencoder = Model(inputs, decoder(encoder(inputs)),name = 'autoencoder')
        print(autoencoder.summary())

        return (encoder, decoder, autoencoder)