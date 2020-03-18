
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
    def build(width=28, height=28, depth=1,intermediate_dim = 256,latent_dim = 32, is_compiled=True):
        # Define encoder model.
        # original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
        # x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
        input_shape = (width, height, depth)
        original_inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
        x = Flatten()(original_inputs)

        x = layers.Dense(intermediate_dim, activation='relu')(x)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()((z_mean, z_log_var))
        encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean,z_log_var,z], name='encoder')


        # Define decoder model.
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        x = layers.Dense(width * height * depth, activation='sigmoid')(x)
        decoder_outputs = Reshape((width, height, depth))(x)
        decoder = tf.keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')

        # Define VAE model.

        z_mean,z_log_var,z = encoder(original_inputs)
        outputs = decoder(z)

        # outputs = outputs.Reshape((width, height, depth))(x)
        vae = tf.keras.Model(inputs=original_inputs, outputs=[outputs,z_mean, z_log_var], name='vae')

        # Add KL divergence regularization loss.
        # kl_loss = - 0.5 * tf.reduce_mean(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        #
        # vae.add_loss(kl_loss)

        # # Calculate custom loss
        # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # vae_loss = K.mean(xent_loss + kl_loss)
        #
        # # Compile
        # vae.add_loss(vae_loss)
        # vae.compile(optimizer='rmsprop')



        # Train.
        # if is_compiled:

        # xent_loss = K.sum(K.binary_crossentropy(original_inputs, outputs), axis=-1)
        # kl_loss = - 0.5 * K.sum(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # vae_loss = K.mean(xent_loss + kl_loss)


        # vae.add_loss(vae_loss)

        # mse = tf.keras.losses.MeanSquaredError()
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # recon_loss = bce(original_inputs, outputs)
        #
        # kl_loss = - 0.5 * tf.reduce_mean(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # vae_loss = recon_loss + kl_loss
        # vae.add_loss(vae_loss)

        # Define VAE Loss
        # @tf.function
        def vae_loss(x_reconstructed, x_true):
            # Reconstruction loss
            encode_decode_loss = x_true * tf.math.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.math.log(1e-10 + 1 - x_reconstructed)
            encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
            # KL Divergence loss
            kl_div_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
            return tf.reduce_mean(encode_decode_loss + kl_div_loss)
        if is_compiled:
            loss_op = vae_loss(original_inputs, outputs)
            vae.add_loss(loss_op)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            vae.compile(optimizer)

        return encoder, decoder, vae

