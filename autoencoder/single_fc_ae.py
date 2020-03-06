
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as prep
import tensorflow.keras.layers as layers
from autoencoder.utility import plot_model_result, load_mnist_dataset, plot_loss


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_dims, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        # self.n_layers = 1
        self.encoder_layer = layers.Dense(n_dims, activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        return self.encoder_layer(inputs)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_dims, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.n_layers = len(n_dims)
        self.decode_middle = layers.Dense(n_dims[0], activation=tf.nn.relu)
        self.recon_layer = layers.Dense(n_dims[1], activation=tf.nn.sigmoid)

    @tf.function
    def call(self, input):
        x = self.decode_middle(input)
        return self.recon_layer(x)

class SingleFullyConnectedAutoencoder(tf.keras.Model):

    def __init__(self,
                 n_dims=[200,32, 784],
                 name = 'autoencoder',
                 **kwargs):
        super(SingleFullyConnectedAutoencoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.encoder = Encoder(n_dims[0])
        self.decoder = Decoder([n_dims[1],n_dims[2]])

    @tf.function
    def call(self, input):
        x = self.encoder(input)
        return self.decoder(x)


def train_single_fcae(num_epochs=5):
    ae = SingleFullyConnectedAutoencoder([200, 32, 784])
    ae.compile(optimizer=tf.optimizers.Adam(0.01),
               loss='binary_crossentropy')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/single_fcae_{epoch}',
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='log')
    ]
    train_data, test_data = load_mnist_dataset(128)
    history = ae.fit(train_data,
                     validation_data=test_data,
                     epochs=num_epochs,
                     callbacks=callbacks)


    # plot histor
    plot_loss(history)


def load_latest_model():
    ae = SingleFullyConnectedAutoencoder([200, 32, 784])
    ae.compile(optimizer=tf.optimizers.Adam(0.01),
               loss='binary_crossentropy')

    # checkpoint_dir = 'checkpoints'
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    latest = 'checkpoints\single_fcae_45'
    ae.load_weights(latest)

    return ae

def test_fc_ae(plot_name=None):
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    ae = load_latest_model()
    decoded_imgs = ae.predict(x_test.reshape(-1, 28 * 28))

    ae.evaluate(x_test.reshape(-1, 28* 28), x_test.reshape(-1,28 * 28))
    plot_model_result(x_test, decoded_imgs, 10, plot_name)

if __name__ == '__main__':
    # train_single_fcae(50)
    test_fc_ae('tmp/signle_fcae_200_32_784_.png')






