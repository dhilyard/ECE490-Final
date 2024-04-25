import numpy as np
import matplotlib.pyplot as plt
import microtorch as t
import microtorch_nn as tnn
import keras

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255  # Normalize pixel values between 0 and 1
X_test = X_test.astype('float32') / 255
y_train = np.eye(10)[y_train.flatten()]  # One-hot encode the labels
y_test = np.eye(10)[y_test.flatten()]

# Flatten the input images
X_train = X_train.reshape(-1, 3072)
X_test = X_test.reshape(-1, 3072)

# Define the loss function
def cross_entropy_loss(yhat, y):
    batch_size = y.shape[0]
    return -(y * t.log(yhat)).sum() / batch_size

# Define the neural network model
model = tnn.Sequential(
    tnn.Linear(3072, 256),
    tnn.ReLU(),
    tnn.Linear(256, 128),
    tnn.ReLU(),
    tnn.Linear(128, 64),
    tnn.ReLU(),
    tnn.Linear(64, 10)
)

# Train the model
def train(model, loss, X_train, y_train, epochs=10, lr=0.001):
    for epoch in range(epochs):
        predicted_labels = model(t.Tensor(X_train))
        loss_value = loss(predicted_labels, t.Tensor(y_train))
        loss_value.backward(1)

        # Update parameters in-place
        for param in model.parameters():
            param.value += -lr * param.grad

        # Reset gradients
        for param in model.parameters():
            param.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value.value:.4f}")

# Train the model
train(model, cross_entropy_loss, X_train, y_train, epochs=10, lr=0.001)

# Evaluate the model on the test set
def evaluate(model, X_test, y_test):
    predicted_labels = model(t.Tensor(X_test))
    predicted_classes = predicted_labels.value.argmax(axis=1)
    true_classes = y_test.argmax(axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

test_accuracy = evaluate(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")