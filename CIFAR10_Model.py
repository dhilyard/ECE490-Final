# This is the process for the model discussed in class for
# reference during the extension of microtorch.

import numpy as np
import matplotlib.pyplot as plt

import microtorch as t
import microtorch_nn as tnn
import DataLoader
import load_cifar_10 as loadcif

import struct
import os
import requests
import gzip


direc = "data/cifar-10-batches-py"
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = loadcif.load_cifar_10_data(direc)
print("Training Info")
print(train_data, train_filenames, train_labels)
print("Testing Info")
print(test_data, test_filenames, test_labels)
print("Label Names")
print(label_names)


# train_dataset = DataLoader.CIFAR_10(root="./data", train=True, download=True)