import matplotlib
matplotlib.use("Agg")

# import the necessary packages

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
# import cv2

from autoencoder.fan_autoencoder import FanAutoencoder

# log
from utils import root_logger
import logging
# import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png",
	help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")

# construct the graph
ap.add_argument("-d", "--n_dims", nargs='+', type=int,
	help="n_dim of layers")
ap.add_argument("-c", "--code_dim", type=int, default=16,
	help="# code_dim - latent layer size ")

# train param
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# epochs  ")

ap.add_argument("-b", "--batch_size", type=int, default=128,
	help="# epochs  ")

args = vars(ap.parse_args())

if 'n_dims' not in args :
    args['n_dims'] = [256,128]
# else:
#     print ( args['n_dims'])

# initialize the number of epochs to train for and batch size
EPOCHS = args['epochs'] #25
BS =  args['batch_size'] #32


# print(args)
root_logger('./logs/log.txt')
logging.info(str(args))

# load the MNIST dataset
logging.info(" loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)  # (sample,w,h) -> (sample, w, h, d)
testX = np.expand_dims(testX, axis=-1)

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# construct our convolutional autoencoder



logging.info(" building autoencoder...")
(encoder, decoder, autoencoder) = FanAutoencoder.build(28, 28, 1, args['n_dims'], args['code_dim'])
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)

# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
logging.info(" making predictions...")
decoded = autoencoder.predict(testX)
outputs = None

# loop over our number of output samples


# for i in range(0, args["samples"]):
# 	# grab the original image and reconstructed image
# 	original = (testX[i] * 255).astype("uint8")
# 	recon = (decoded[i] * 255).astype("uint8")
#
# 	# stack the original and reconstructed image side-by-side
# 	output = np.hstack([original, recon])
#
# 	# if the outputs array is empty, initialize it as the current
# 	# side-by-side image display
# 	if outputs is None:
# 		outputs = output
#
# 	# otherwise, vertically stack the outputs
# 	else:
# 		outputs = np.vstack([outputs, output])

# save the outputs image to disk
# cv2.imwrite(args["output"], outputs)

plt.figure(figsize=(20, 4))
n = args["samples"]
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    original = (testX[i].reshape(28, 28) * 255).astype("uint8")
    recon =  (decoded[i].reshape(28, 28) * 255).astype("uint8")

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
    plt.savefig(args["output"])
