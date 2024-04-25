# Extensions for microtorch

import numpy as np

# Defining the softmax function
def softmax(values):
    # Computing element wise exponential value
    exp_values = np.exp(values)

    # Computing sum of these values
    exp_values_sum = np.sum(exp_values)

    # Returing the softmax output.
    return exp_values / exp_values_sum

# Cross Entropy function.
def cross_entropy_loss(y_pred, y_true):
    # computing softmax values for predicted values
    y_pred = softmax(y_pred)
    loss = 0

    # Doing cross entropy Loss
    for i in range(len(y_pred)):
        # Here, the loss is computed using the
        # above mathematical formulation.
        loss = loss + (-1 * y_true[i] * np.log(y_pred[i]))

    return loss