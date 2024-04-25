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

# Reshape the input data for CNN
X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

# Define the loss function
def cross_entropy_loss(yhat, y):
    batch_size = y.shape[0]
    return -(y * t.log(yhat)).sum() / batch_size

# Define the Conv2d layer
class Conv2d(tnn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = tnn.Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = tnn.Parameter(np.zeros(out_channels))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        out_height = (height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_width) // self.stride + 1
        x_padded = np.pad(x.value, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for out_channel in range(self.out_channels):
            for height_stride in range(out_height):
                for width_stride in range(out_width):
                    input_slice = x_padded[:, :, height_stride * self.stride:height_stride * self.stride + kernel_height,
                                   width_stride * self.stride:width_stride * self.stride + kernel_width]
                    kernel_slice = self.weight.value[out_channel]
                    output[:, out_channel, height_stride, width_stride] = (input_slice * kernel_slice).sum(axis=(1, 2, 3)) + self.bias.value[out_channel]

        return t.Tensor(output, parents=[t.Tensor(x), self.weight, self.bias], op=self)

# Define the MaxPool2d layer
class MaxPool2d(tnn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        out_height = (height - kernel_height) // self.stride + 1
        out_width = (width - kernel_width) // self.stride + 1
        output = np.zeros((batch_size, channels, out_height, out_width))

        for batch in range(batch_size):
            for channel in range(channels):
                for height_stride in range(out_height):
                    for width_stride in range(out_width):
                        input_slice = x.value[batch, channel, height_stride * self.stride:height_stride * self.stride + kernel_height,
                                                width_stride * self.stride:width_stride * self.stride + kernel_width]
                        output[batch, channel, height_stride, width_stride] = input_slice.max()

        return t.Tensor(output, parents=[t.Tensor(x)], op=self)

class Flatten(tnn.Module):
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_reshaped = x.value.reshape(batch_size, -1)
        return t.Tensor(x_reshaped, parents=[t.Tensor(x)], op=self)

# Define the optimizer
class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {param: np.zeros_like(param.value) for param in parameters}
        self.v = {param: np.zeros_like(param.value) for param in parameters}
        self.t = 0

    def step(self):
        self.t += 1
        for param in self.parameters:
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * param.grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# Define the neural network model
model = tnn.Sequential(
    Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    tnn.ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    tnn.ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    tnn.Linear(32 * 8 * 8, 128),
    tnn.ReLU(),
    tnn.Linear(128, 10)
)

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Train the model
def train(model, loss, optimizer, X_train, y_train, batch_size=64, epochs=10):
    num_batches = X_train.shape[0] // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            predicted_labels = model(t.Tensor(X_batch))
            loss_value = loss(predicted_labels, t.Tensor(y_batch))
            
            # Initialize gradient with respect to the loss
            grad = t.Tensor(np.ones_like(loss_value.value))
            
            loss_value.backward(grad)  # Pass the gradient to the backward method

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss_value.value

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Train the model
train(model, cross_entropy_loss, optimizer, X_train, y_train, batch_size=64, epochs=10)

# Evaluate the model on the test set
def evaluate(model, X_test, y_test):
    predicted_labels = model(t.Tensor(X_test))
    predicted_classes = predicted_labels.value.argmax(axis=1)
    true_classes = y_test.argmax(axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

test_accuracy = evaluate(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")