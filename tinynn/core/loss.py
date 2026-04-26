"""Loss functions"""

import numpy as np
from tinynn.utils.math import log_softmax
from tinynn.utils.math import sigmoid
from tinynn.utils.math import softmax


class Loss:

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class MSE(Loss):

    def loss(self, predictions, targets):
        pass

    def grad(self, predictions, targets):
        pass


class MAE(Loss):

    def loss(self, predictions, targets):
        pass

    def grad(self, predictions, targets):
        pass


class Huber(Loss):

    def __init__(self, delta=1.0):
        self._delta = delta

    def loss(self, predictions, targets):
        pass

    def grad(self, predictions, targets):
        pass


class SoftmaxCrossEntropy(Loss):

    def __init__(self, T=1.0, weights=None):
        self._weights = np.asarray(weights) if weights is not None else weights
        self._T = T

    def loss(self, logits, labels):
        pass

    def grad(self, logits, labels):
        pass


class SigmoidCrossEntropy(Loss):
    """let logits = a, label = y, weights[neg] = w1, weights[pos] = w2
    L = - w2 * y * log(1 / (1 + exp(-a)) - w1 * (1-y) * log(exp(-a) / (1 + exp(-a))
      = w1 * a * (1 - y) - (w2 * y - w1 * (y - 1)) * log(sigmoid(a))
    if w1 == w2 == 1:
    L = a * (1 - y) - log(sigmoid(a))

    G = w1 * sigmoid(a) - w2 * y + (w2 - w1) * y * sigmoid(a)
    if w1 == w2 == 1:
    G = sigmoid(a) - y
    """
    def __init__(self, weights=None):
        weights = np.ones(2, dtype=np.float32) if weights is None else weights
        self._weights = np.asarray(weights)

    def loss(self, logits, labels):
        pass

    def grad(self, logits, labels):
        pass
