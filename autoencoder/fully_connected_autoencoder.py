import tensorflow as tf
import matplotlib.pyplot as plt


class FullyConnectedAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(FullyConnectedAutoEncoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        
        
        self.bottleneck = tf.keras.layers.Dense(16, activation=tf.nn.relu)
    
        self.dense4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        
        self.dense_final = tf.keras.layers.Dense(784)
        
    @tf.function
    def call(self, inp):
        x_reshaped = self.flatten_layer(inp)
        x = self.dense1(x_reshaped)
        x = self.dense2(x)
        x = self.bottleneck(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense_final(x)
        return x
    


## eager mode train

def loss(x, x_bar):
#     return tf.losses.mean_squared_error(x, x_bar)
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(x, x_bar)))
    return reconstruction_error

def grad(model, inputs):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(inputs, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs, reconstruction



def test_eagermode_training(num_epochs = 5,batch_size = 128):
    # num_epochs = 5
    # batch_size = 128
    train_data, _ = load_mnist_dataset(batch_size)
    model = FullyConnectedAutoEncoder()
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}')
        for step, (x_batch ,_)in enumerate(train_data):
            loss_value, grads, inputs_reshaped, reconstruction = grad(model, x_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))           
            if step % 200 == 0:
                print("Step: {},         Loss: {}".format(step,
                                                        loss(inputs_reshaped, reconstruction).numpy()))


def load_mnist_dataset(batch_size):
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.
    train_data = tf.data.Dataset.from_tensor_slices((x_train,x_train)).batch(batch_size).shuffle(buffer_size=1024)
    test_data = tf.data.Dataset.from_tensor_slices((x_test,x_test)).batch(batch_size).shuffle(buffer_size=512)
    return train_data, test_data

# fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
# validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
# sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

def test_graph_mode_training(num_epochs = 50,batch_size = 128):

    train_data, test_data = load_mnist_dataset(batch_size)

    ae = FullyConnectedAutoEncoder()
    # plot the model not fit the plan, need to study
    # tf.keras.utils.plot_model(ae, 'multi_input_and_output_model.png', show_shapes=True)

    ae.compile(
        optimizer = tf.optimizers.Adam(0.01),
        # loss ='categorical_crossentropy'
        loss = 'binary_crossentropy'
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/FullyConnectedAutoEncoder_{epoch}',
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='log')
    ]


    history = ae.fit(train_data,
           # batch_size = 128,
           # steps_per_epoch=30,
           validation_data= test_data,
           epochs = num_epochs,
           callbacks=callbacks)


def plot_model_result(input_imgs, decoded_imgs, n, savename=None):
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(input_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if  savename:
        plt.savefig(savename)
    plt.show()

def load_latest_model():
    ae = FullyConnectedAutoEncoder()
    ae.compile(
        optimizer = tf.optimizers.Adam(0.01),
        # loss ='categorical_crossentropy'
        loss = 'binary_crossentropy'
    )
    checkpoint_dir = 'checkpoints'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    ae.load_weights(latest)

    # keral need to build to summary
    # ae.build(input_shape=(28,28))
    # ae.summary()
    return ae


def test_fc_ae(plot_name=None):

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # train_data, test_data = load_dataset(128)

    # x_test = test_data[0]
    ae = load_latest_model()
    decoded_imgs = ae.predict(x_test)

    ae.evaluate(x_test, x_test.reshape(-1,28 * 28))
    plot_model_result(x_test, decoded_imgs, 10, plot_name)


if __name__ == '__main__':
    # test_eagermode_training()

    # test_graph_mode_training(100)


    test_fc_ae('tmp/acae_1.png')

    # ae.eva

