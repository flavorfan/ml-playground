import tensorflow as tf
import matplotlib.pyplot as plt


def load_mnist_dataset(batch_size=128):
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.
    train_data = tf.data.Dataset.from_tensor_slices((x_train,x_train)).batch(batch_size).shuffle(buffer_size=1024)
    test_data = tf.data.Dataset.from_tensor_slices((x_test,x_test)).batch(batch_size).shuffle(buffer_size=512)
    return train_data, test_data


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

def plot_loss(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();