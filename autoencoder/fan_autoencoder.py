
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers
from tensorflow.keras import backend as K

# def loss(y_true, y_pred):
#     ce = tf.losses.mse
#     ce = 0.5 *
#     return ce


class FanAutoencoder:
    @staticmethod
    def build(width, height, depth, n_dims=[256,128], code_dim=16, is_compiled=True):
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

        if is_compiled:
            opt = Adam(lr=1e-3)
            autoencoder.compile(loss="mse", optimizer=opt)

        return (encoder, decoder, autoencoder)


# 另外一种实现
# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=K.shape(z_mean))
#     return z_mean + K.exp(z_log_var / 2) * epsilon
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class FanVariationalAutoEncoder():
    @staticmethod
    # def build(width=28, height=28, depth=1,intermediate_dim = 256,latent_dim = 32, is_compiled=True):
    def build(original_dim = 784, intermediate_dim=512, latent_dim=20, is_compiled=True):
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
        # outputs = layers.Dense(original_dim, activation='sigmoid')(x)
        outputs = layers.Dense(original_dim)(x)  # fanck 0320 remove the sigmoid
        decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

        # Define VAE model.
        vae_outputs = decoder(z)

        # outputs = outputs.Reshape((width, height, depth))(x)
        vae = tf.keras.Model(inputs=original_inputs, outputs=[vae_outputs,z_mean, z_log_var], name='vae')

        return encoder, decoder, vae

