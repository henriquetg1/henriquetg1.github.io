"""Checks the by-hand numbers of Exercise 1 and the MLP gradients.

Exercise 1 itself is worked by hand in the report. This script only confirms
that the hand calculation and mlp.py agree, then compares the backward pass
against finite differences for every loss.
"""

import numpy as np

from mlp import MLP, SigmoidBCE, SoftmaxCCE, TanhMSE, one_hot


def exercise_1():
    x = np.array([[0.5, -0.2]])
    y = np.array([[1.0]])
    net = MLP([2, 2, 1], TanhMSE(), np.random.default_rng(42))
    net.W = [np.array([[0.3, -0.1], [0.2, 0.4]]), np.array([[0.5, -0.3]])]
    net.b = [np.array([0.1, -0.2]), np.array([0.2])]

    u, inputs = net.forward(x)
    h, y_hat = inputs[1], net.loss.output(u)
    print("z1      ", np.arctanh(h).round(6))
    print("h1      ", h.round(6))
    print("u2      ", u.round(6))
    print("y_hat   ", y_hat.round(6))
    print("L       ", round(net.loss.value(u, y), 6))
    print("dL/dy   ", (2 * (y_hat - y)).round(6))
    print("dy/du   ", (1 - y_hat**2).round(6))
    print("dh/dz   ", (1 - h**2).round(6))

    grads_W, grads_b = net.backward(u, inputs, y)
    print("dL/dW2  ", grads_W[1].round(6))
    print("dL/db2  ", grads_b[1].round(6))
    print("dL/dh1  ", (grads_b[1] * net.W[1]).round(6))
    print("dL/dW1  ", grads_W[0].round(6).tolist())
    print("dL/db1  ", grads_b[0].round(6))
    squared_norm = sum((g**2).sum() for g in grads_W + grads_b)
    print("|grad|^2", round(squared_norm, 6))

    net.update(grads_W, grads_b, lr=0.3)
    print("W2 new  ", net.W[1].round(6))
    print("b2 new  ", net.b[1].round(6))
    print("W1 new  ", net.W[0].round(6).tolist())
    print("b1 new  ", net.b[0].round(6))
    u_new, inputs_new = net.forward(x)
    print("h1 new  ", inputs_new[1].round(6))
    print("u2 new  ", u_new.round(6))
    print("y_hat new", net.loss.output(u_new).round(6))
    print("L after ", round(net.loss.value(u_new, y), 6))


def worst_gradient_error(net, X, Y, step=1e-6):
    """Largest gap between the backward pass and central finite differences."""
    u, inputs = net.forward(X)
    grads_W, grads_b = net.backward(u, inputs, Y)
    worst = 0.0
    for params, grads in zip(net.W + net.b, grads_W + grads_b):
        for i in np.ndindex(params.shape):
            original = params[i]
            params[i] = original + step
            up = net.loss.value(net.forward(X)[0], Y)
            params[i] = original - step
            down = net.loss.value(net.forward(X)[0], Y)
            params[i] = original
            worst = max(worst, abs((up - down) / (2 * step) - grads[i]))
    return worst


def finite_differences():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    binary = rng.integers(0, 2, size=(20, 1)).astype(float)
    multi = one_hot(rng.integers(0, 3, size=20), 3)

    cases = [
        ("TanhMSE    [4, 5, 1]", [4, 5, 1], TanhMSE(), binary),
        ("SigmoidBCE [4, 5, 1]", [4, 5, 1], SigmoidBCE(), binary),
        ("SoftmaxCCE [4, 5, 3]", [4, 5, 3], SoftmaxCCE(), multi),
        ("SoftmaxCCE [4, 5, 4, 3]", [4, 5, 4, 3], SoftmaxCCE(), multi),
    ]
    for name, sizes, loss, Y in cases:
        error = worst_gradient_error(MLP(sizes, loss, rng), X, Y)
        print(f"{name:26s} worst gradient error: {error:.2e}")


if __name__ == "__main__":
    exercise_1()
    print()
    finite_differences()
