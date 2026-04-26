"""Model class manage the network, loss function and optimizer."""

import pickle


class Model:

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        pass

    def backward(self, predictions, targets):
        pass

    def apply_grads(self, grads):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    @property
    def is_training(self):
        pass

    @is_training.setter
    def is_training(self, is_training):
        pass
