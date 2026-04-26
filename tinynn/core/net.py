"""Feed-forward Neural Network class."""

import copy

from tinynn.utils.structured_param import StructuredParam


class Net:

    def __init__(self, layers):
        self.layers = layers
        self._is_training = True

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def forward(self, inputs):
        pass

    def backward(self, grad):
        # back propagation
        pass

    @property
    def params(self):
        pass

    @params.setter
    def params(self, params):
        pass

    @property
    def is_training(self):
        pass

    @is_training.setter
    def is_training(self, is_training):
        pass
