# This is the process for the model discussed in class for
# reference during the extension of microtorch.

import numpy as np
import matplotlib.pyplot as plt

import microtorch as t
import microtorch_nn as tnn

import struct
import os
import requests
import gzip

def feature_n_pxls(imgs):
    n, *shape = imgs.shape
    return np.sum(imgs[:, :, :].reshape(n, -1) > 128, axis=1)

def feature_y_var(imgs):
    wts = imgs.mean(axis=-2)
    mean = (np.arange(imgs.shape[-2]) * wts).sum(axis=-1) / wts.sum(axis=-1)
    var = ((np.arange(imgs.shape[-2]) - mean[:, None])**2 * wts).sum(axis=-1) / wts.sum(axis=-1)
    return var

zero_one_train_features = np.load('zero_one_train_features.npz')
print(zero_one_train_features)
FEATURE_MEAN = zero_one_train_features['mean']
FEATURE_STD = zero_one_train_features['std']
features = zero_one_train_features['normed_features']
print(features)
labels = zero_one_train_features['labels']
print(labels)

def feature_extraction(imgs):
    features = np.stack((feature_n_pxls(imgs),
                     feature_y_var(imgs)), axis=-1)
    return (features - FEATURE_MEAN) / FEATURE_STD


def loss(predicted_labels, true_labels):
    # Make sure predicted_labels and true_labels have same shape
    ### BEGIN SOLUTION
    y = true_labels[..., None]
    yhat = predicted_labels
    assert y.shape == yhat.shape
    return t.maximum(- y * yhat, 0).sum()  # / y.shape[-1]
    ### END SOLUTION


# Define model = ?
### BEGIN SOLUTION
model = tnn.Sequential(
    tnn.Linear(2, 5),
    tnn.ReLU(),
    tnn.Linear(5, 1))

first_linear = tnn.Linear(2, 5)
first_activation = tnn.ReLU()
second_linear = tnn.Linear(5, 1)

model2 = tnn.Sequential(first_linear, first_activation, second_linear)




### END SOLUTION

def train_by_gradient_descent(model, loss, train_features, train_labels, lr=0.0001):
    train_features_tensor = t.Tensor(train_features)
    predicted_labels = model(train_features_tensor)
    # print(predicted_labels)

    loss_t = loss(predicted_labels, train_labels)
    loss_t.backward(1)
    loss_t_minus_1 = 2 * loss_t.value  # Fake  value to make the while test pass once
    niter = 0
    while np.abs(loss_t.value - loss_t_minus_1) / loss_t.value > 0.01:  # Stopping criterion
        for param in model.parameters():
            assert param.grad is not None
            # print("before:", id(param))
            param.value = param.value - lr * param.grad  # Gradient descent
            # print("after:", id(param))
        loss_t.zero_grad()
        # Recompute the gradients
        predicted_labels = model(train_features_tensor)
        loss_t_minus_1 = loss_t.value
        loss_t = loss(predicted_labels, train_labels)
        loss_t.backward(1)  # Compute gradients for next iteration

        # If loss increased, decrease lr. Works for gradient descent, not for stochatic gradient descent.
        if loss_t.value > loss_t_minus_1:
            lr = lr / 2

        ### DEBUGing information
        iswrong = (train_labels * predicted_labels.value.ravel()) < 0
        misclassified = (iswrong).sum() / iswrong.shape[0]
        print(f"loss: {loss_t.value:04.04f}, delta loss: {loss_t.value - loss_t_minus_1:04.04f},"
              f"train misclassified: {misclassified:04.04f}")
        if niter % 20 == 0:  # plot every 20th iteration
            fig, ax = plt.subplots(1, 1)

        niter += 1
    return model


trained_model = train_by_gradient_descent(model, loss, features, labels)
# Load MNIST dataset from uint8 byte files


# Define the URLs and file names
url_images = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
url_labels = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
image_file = "data/t10k-images-idx3-ubyte"
label_file = "data/t10k-labels-idx1-ubyte"

# Create the directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download and decompress the image file
if not os.path.exists(image_file):
    r = requests.get(url_images)
    with open(image_file + ".gz", "wb") as f:
        f.write(r.content)
    with gzip.open(image_file + ".gz", "rb") as f_in:
        with open(image_file, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(image_file + ".gz")

# Download and decompress the label file
if not os.path.exists(label_file):
    r = requests.get(url_labels)
    with open(label_file + ".gz", "wb") as f:
        f.write(r.content)
    with gzip.open(label_file + ".gz", "rb") as f_in:
        with open(label_file, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(label_file + ".gz")


# Ref:https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
def mnist_read_labels(fname='data/train-labels-idx1-ubyte'):
    with open(fname, 'rb') as file:
        # The file starts with 4 byte 2 unsigned ints
        magic, size = struct.unpack('>II', file.read(8))
        assert magic == 2049
        labels = np.frombuffer(file.read(), dtype='u1')
        return labels


# Ref:https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
def mnist_read_images(fname='data/train-images-idx3-ubyte'):
    with open(fname, 'rb') as file:
        # The file starts with 4 byte 4 unsigned ints
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        assert magic == 2051
        image_data = np.frombuffer(file.read(), dtype='u1')
        images = image_data.reshape(size, rows, cols)
        return images


test_images = mnist_read_images('data/t10k-images-idx3-ubyte')
test_labels = mnist_read_labels('data/t10k-labels-idx1-ubyte')
zero_one_filter = (test_labels == 0) | (test_labels == 1)
zero_one_test_images = test_images[zero_one_filter, ...]
zero_one_test_labels = test_labels[zero_one_filter, ...]


def returnclasslabel(test_imgs):
    Xtest = feature_extraction(test_imgs)
    return np.where(
        trained_model(Xtest).value.ravel() > 0,
        0,
        1)


zero_one_predicted_labels = returnclasslabel(zero_one_test_images)

# Find test_accuracy = ?
### BEGIN SOLUTION
test_accuracy = np.sum(zero_one_test_labels == zero_one_predicted_labels) / len(zero_one_test_labels)

### END SOLUTION
print(test_accuracy)
assert test_accuracy > 0.90