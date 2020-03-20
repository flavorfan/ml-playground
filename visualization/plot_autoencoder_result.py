import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import plotly.express as px


def training_plot(H,filename, epochs):
    # construct a plot that plots and saves the training history
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(filename)

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

# 10 x 10 grid img
def grid_img(x_concat, filename):
    index = 0
    new_im = Image.new('L', (280, 280))
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = x_concat[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(filename)
    # plt.imshow(np.asarray(new_im))
    # plt.show()