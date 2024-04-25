# This is the process for the model discussed in class for
# reference during the extension of microtorch.

import numpy as np
import matplotlib.pyplot as plt

import microtorch as t
import microtorch_nn as tnn
import microtorch_ext as tx
import DataLoader


# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Use Matplotlib to check images, takes numpy array
# If it has 3 channels use a library called Pillow to read from arrays.
# Focus on multi-class for cross-entropy-loss
# pre-process to have 0 mean
# implement softmax after output from 2 layer neural network, pass 10 of the softmax outputs through cross-entropy-loss

# My DataLoader matches all the data it should from PyTorch
train_dataset = DataLoader.CIFAR_10(root="./data", train=True, download=True)
test_dataset = DataLoader.CIFAR_10(root="./data", train=False, download=True)

train_features = train_dataset.__dict__.get('data') / 255

print(tx.softmax(train_features))


# Image Printing

import _pickle as pickle
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

cifar10 = "./data/cifar-10-batches-py/"

parser = argparse.ArgumentParser("Plot training images in cifar10 dataset")
parser.add_argument("-i", "--image", type=int, default=2,
                    help="Index of the image in cifar10. In range [0, 49999]")
args = parser.parse_args()

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def cifar10_plot(data, meta, im_idx=0):
    im = data['data'][im_idx, :]

    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    print("shape: ", img.shape)
    print("label: ", data['labels'][im_idx])
    print("category:", meta['label_names'][data['labels'][im_idx]])

    plt.imshow(img)
    plt.show()


def main():
    batch = (args.image // 10000) + 1
    idx = args.image - (batch - 1) * 10000

    data = unpickle(os.path.join(cifar10, "data_batch_" + str(batch)))
    meta = unpickle(os.path.join(cifar10, "batches.meta"))
    cifar10_plot(data, meta, im_idx=idx)


main()