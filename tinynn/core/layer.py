"""Network layers."""

import numpy as np
from tinynn.core.initializer import Ones
from tinynn.core.initializer import XavierUniform
from tinynn.core.initializer import Zeros
from tinynn.utils.math import sigmoid


def empty(shape, dtype=np.float32):
    pass


class Layer:
    """Base class for layers."""

    def __init__(self):
        self.params = {p: None for p in self.param_names}
        self.nt_params = {p: None for p in self.nt_param_names}
        self.initializers = {}

        self.grads = {}
        self.shapes = {}

        self._is_training = True  # used in BatchNorm/Dropout layers
        self._is_init = False

        self.ctx = {}

    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return f"layer: {self.name}\tshape: {shape}"

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    @property
    def is_init(self):
        pass

    @is_init.setter
    def is_init(self, is_init):
        pass

    @property
    def is_training(self):
        pass

    @is_training.setter
    def is_training(self, is_train):
        pass

    @property
    def name(self):
        pass

    @property
    def param_names(self):
        pass

    @property
    def nt_param_names(self):
        pass

    def _init_params(self):
        pass


class Dense(Layer):
    """A dense layer operates `outputs = dot(intputs, weight) + bias`
    :param num_out: A positive integer, number of output neurons
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    """
    def __init__(self,
                 num_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [None, num_out], "b": [num_out]}

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass

    @property
    def param_names(self):
        pass


class Conv2D(Layer):
    """Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (height, width,
        in_channels, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": self.kernel_shape, "b": self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        """Accelerate convolution via im2col trick.
        An example (assuming only one channel and one filter):
         input = | 43  16  78 |         kernel = | 4  6 |
          (X)    | 34  76  95 |                  | 7  9 |
                 | 35   8  46 |

        After im2col and kernel flattening:
         col  = | 43  16  34  76 |     kernel = | 4 |
                | 16  78  76  95 |      (W)     | 6 |
                | 34  76  35   8 |              | 7 |
                | 76  95   8  46 |              | 9 |
        """
        pass

    def backward(self, grad):
        """Compute gradients w.r.t. layer parameters and backward gradients.
        :param grad: gradients from previous layer
            with shape (batch_sz, out_h, out_w, out_c)
        :return d_in: gradients to next layers
            with shape (batch_sz, in_h, in_w, in_c)
        """
        pass

    def _inputs_preprocess(self, inputs):
        pass

    def _grads_postprocess(self, grads):
        pass

    @property
    def param_names(self):
        pass


class ConvTranspose2D(Conv2D):

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__(kernel, stride, padding, w_init, b_init)
        self.origin_stride = stride
        self.stride = (1, 1)

    def _inputs_preprocess(self, inputs):
        pass

    def _grads_postprocess(self, grads):
        pass

    @staticmethod
    def _insert_zeros(inputs, s_h, s_w, mode):
        pass


class MaxPool2D(Layer):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=None,
                 padding="VALID"):
        """Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.kernel_shape = pool_size
        self.stride = stride if stride is not None else pool_size

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass


class RNN(Layer):

    def __init__(self,
                 num_hidden,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()
        self.n_h = num_hidden
        self.initializers = {"W": w_init, "b": b_init}


    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass

    @property
    def param_names(self):
        pass


class LSTM(Layer):

    def __init__(self,
                 num_hidden,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()
        self.n_h = num_hidden
        self.initializers = {"W_g": w_init, "W_c": w_init,
                             "b_g": b_init, "b_c": b_init}

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass

    @property
    def param_names(self):
        pass


class BatchNormalization(Layer):

    def __init__(self,
                 momentum=0.99,
                 gamma_init=Ones(),
                 beta_init=Zeros(),
                 epsilon=1e-5):
        super().__init__()
        self.m = momentum
        self.epsilon = epsilon

        self.initializers = {"gamma": gamma_init, "beta": beta_init}
        self.reduce = None

    def forward(self, inputs):
        # self.reduce = (0,) if inputs.ndim == 2 else (0, 1, 2)
        pass

    def backward(self, grad):
        # grads w.r.t. params
        pass

    @property
    def param_names(self):
        pass

    @property
    def nt_param_names(self):
        pass


class Reshape(Layer):

    def __init__(self, *output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass


class Flatten(Reshape):

    def __init__(self):
        super().__init__(-1)


class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__()
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass


class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class Softplus(Activation):

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class Tanh(Activation):

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class ReLU(Activation):

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class LeakyReLU(Activation):

    def __init__(self, slope=0.2):
        super().__init__()
        self._slope = slope

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class GELU(Activation):
    """Gaussian Error Linear Units
    ref: https://arxiv.org/pdf/1606.08415.pdf
    """

    def __init__(self):
        super().__init__()
        self._alpha = 0.1702
        self._cache = None

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class ELU(Activation):

    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def func(self, x):
        pass

    def derivative(self, x):
        pass


def im2col(img, k_h, k_w, s_h, s_w):
    """Transform padded image into column matrix.
    :param img: padded inputs of shape (B, in_h, in_w, in_c)
    :param k_h: kernel height
    :param k_w: kernel width
    :param s_h: stride height
    :param s_w: stride width
    :return col: column matrix of shape (B*out_h*out_w, k_h*k_h*inc)
    """
    pass


def get_padding_2d(in_shape, k_shape, mode):

    pass
