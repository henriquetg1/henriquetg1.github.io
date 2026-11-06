"""A multi-layer perceptron written from scratch with NumPy.

Activations, losses, forward pass, gradients and the update step are all
hand-written here. The same MLP class serves Exercises 2, 3 and 4: the layer
sizes and the loss are constructor arguments, so forward, backward and update
never change between exercises.

Notation follows the statement. For layer l: z = W h_prev + b, h = tanh(z).
The last layer stops at its pre-activation u; the loss object turns u into the
network output (sigmoid, softmax or tanh) and knows dL/du.
"""

import numpy as np

EPS = 1e-12  # keeps log() finite when a probability saturates


class SigmoidBCE:
    """Sigmoid output + binary cross-entropy (Exercise 2)."""

    def output(self, u):
        return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))

    def value(self, u, y):
        p = np.clip(self.output(u), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def gradient(self, u, y):
        # dL/du = (p - y) / N, summed over the batch by the caller's matrix product
        return (self.output(u) - y) / len(y)

    def predict(self, u):
        return (u[:, 0] > 0).astype(int)


class SoftmaxCCE:
    """Softmax output + categorical cross-entropy (Exercises 3 and 4)."""

    def output(self, u):
        # Subtracting the row maximum avoids overflow and leaves softmax unchanged
        e = np.exp(u - u.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def value(self, u, y):
        p = np.clip(self.output(u), EPS, 1.0)
        return -np.mean(np.sum(y * np.log(p), axis=1))

    def gradient(self, u, y):
        # Same form as the binary case: dL/du = (p - y) / N
        return (self.output(u) - y) / len(y)

    def predict(self, u):
        return u.argmax(axis=1)


class TanhMSE:
    """Tanh output + squared error (only used to check Exercise 1)."""

    def output(self, u):
        return np.tanh(u)

    def value(self, u, y):
        return np.mean((y - self.output(u)) ** 2)

    def gradient(self, u, y):
        y_hat = self.output(u)
        return 2 * (y_hat - y) * (1 - y_hat**2) / len(y)

    def predict(self, u):
        return (u[:, 0] > 0).astype(int)


class MLP:
    """Fully connected network with tanh hidden layers.

    layer_sizes: e.g. [2, 8, 1] is 2 inputs, one hidden layer of 8, 1 output.
    loss:        SigmoidBCE, SoftmaxCCE or TanhMSE.
    rng:         numpy Generator used to draw the initial weights.
    """

    def __init__(self, layer_sizes, loss, rng):
        self.layer_sizes = list(layer_sizes)
        self.loss = loss
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier initialization, suited to tanh; biases start at zero
            std = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(rng.normal(0.0, std, size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        """Return the output pre-activation u and every layer's input."""
        inputs = [X]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            inputs.append(np.tanh(inputs[-1] @ W.T + b))
        u = inputs[-1] @ self.W[-1].T + self.b[-1]
        return u, inputs

    def backward(self, u, inputs, Y):
        """Backpropagation: return dL/dW and dL/db for every layer."""
        n_layers = len(self.W)
        grads_W, grads_b = [None] * n_layers, [None] * n_layers
        dL_dz = self.loss.gradient(u, Y)  # one row per sample
        for l in reversed(range(n_layers)):
            h_prev = inputs[l]
            grads_W[l] = dL_dz.T @ h_prev  # z = W h_prev + b, so dz/dW = h_prev
            grads_b[l] = dL_dz.sum(axis=0)  # dz/db = 1
            if l > 0:
                dL_dh = dL_dz @ self.W[l]  # chain through z = W h_prev + b
                dL_dz = dL_dh * (1 - h_prev**2)  # chain through h_prev = tanh(z_prev)
        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        """One step of gradient descent on every weight and bias."""
        for l in range(len(self.W)):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    def fit(self, X, Y, lr, epochs):
        """Full-batch gradient descent. Return the training loss of each epoch."""
        history = np.zeros(epochs)
        for epoch in range(epochs):
            u, inputs = self.forward(X)
            history[epoch] = self.loss.value(u, Y)
            grads_W, grads_b = self.backward(u, inputs, Y)
            self.update(grads_W, grads_b, lr)
        return history

    def predict(self, X):
        u, _ = self.forward(X)
        return self.loss.predict(u)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]
