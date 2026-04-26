"""Various optimization algorithms and learning rate schedulers."""

import numpy as np


class Optimizer:

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        # compute the gradient step
        pass

    def compute_step(self, grads):
        pass

    def _compute_step(self, grads):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def _compute_step(self, grads):
        pass


class Adam(Optimizer):

    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

    def _compute_step(self, grads):
        pass


class RAdam(Optimizer):
    """Rectified Adam. Ref: https://arxiv.org/pdf/1908.03265v1.pdf """
    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._b1 = beta1
        self._b2 = beta2
        self._epsilon = epsilon

        self._t = 0
        self._m = 0
        self._v = 0

        self.rho = 2.0 / (1 - self._b2) - 1.0

    def _compute_step(self, grads):
        pass


class RMSProp(Optimizer):
    """Root Mean Square Prop optimizer
    mean_square = decay * mean_square{t-1} + (1-decay) * grad_t**2
    mom = momentum * mom{t-1} + lr * grad_t / sqrt(mean_square + epsilon)
    """
    def __init__(self,
                 lr=0.01,
                 decay=0.99,
                 momentum=0.0,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._rho = decay
        self._momentum = momentum
        self._epsilon = epsilon

        self._rms = 0
        self._mom = 0

    def _compute_step(self, grads):
        pass


class Momentum(Optimizer):
    """accumulation = momentum * accumulation + gradient
    variable -= learning_rate * accumulation
    """
    def __init__(self, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._momentum = momentum
        self._acc = 0

    def _compute_step(self, grads):
        pass


class Adagrad(Optimizer):
    """AdaGrad optimizer
    accumulation = - (learning_rate / sqrt(G + epsilon)) * gradient
    where G is the element-wise sum of square gradient
    ref: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, lr, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self._g = 0
        self._epsilon = epsilon

    def _compute_step(self, grads):
        pass


class Adadelta(Optimizer):
    """Adadelta algorithm (https://arxiv.org/abs/1212.5701)"""
    def __init__(self, lr=1.0, decay=0.9, epsilon=1e-8, weight_decay=0.0,):
        super().__init__(lr, weight_decay)
        self._epsilon = epsilon
        self._rho = decay
        self._rms = 0  # running average of square gradient
        self._delta = 0  # running average of delta

    def _compute_step(self, grads):
        pass


class BaseScheduler:
    """BaseScheduler model receive a optimizer and Adjust the lr
    by calling step() method during training.
    """
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._init_lr = self.curr_lr

        self._t = 0

    def step(self):
        pass

    def _compute_lr(self):
        raise NotImplementedError

    @property
    def curr_lr(self):
        pass


class StepLR(BaseScheduler):
    """LR decayed by gamma every "step_size" epochs."""
    def __init__(self,
                 optimizer,
                 step_size,
                 gamma=0.1):
        super().__init__(optimizer)
        assert step_size >= 1
        self._step_size = step_size
        self._gamma = gamma

    def _compute_lr(self):
        pass


class MultiStepLR(BaseScheduler):
    """LR decayed by gamma when #steps reaches one of the milestones.
    Milestones must be monotonically increasing.
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        milestones = [int(m) for m in milestones]
        assert len(milestones) > 0, "milestones requires at-least one element!"
        assert all(x < y for x, y in zip(milestones[:-1], milestones[1:])), \
               "milestones must be a list of int and be increasing!"

        self._milestones = milestones
        self._gamma = gamma

    def _compute_lr(self):
        pass


class ExponentialLR(BaseScheduler):
    """ExponentialLR is computed by:
    lr_decayed = lr * decay_rate ^ (current_steps / decay_steps)
    """
    def __init__(self,
                 optimizer,
                 decay_steps,
                 decay_rate=(1. / np.e)):
        super().__init__(optimizer)
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

    def _compute_lr(self):
        pass


class LinearLR(BaseScheduler):
    """Linear decay learning rate when the number of the epoch is in
    [start_step, start_step + decay_steps]
    """
    def __init__(self,
                 optimizer,
                 decay_steps,
                 final_lr=1e-6,
                 start_step=0):
        super().__init__(optimizer)
        assert decay_steps > 0

        self._lr_delta = (final_lr - self._init_lr) / decay_steps

        self._final_lr = final_lr
        self._decay_steps = decay_steps
        self._start_step = start_step

    def _compute_lr(self):
        pass


class CyclicalLR(BaseScheduler):
    """Cyclical increase and decrease learning rate within a reasonable range.
    Ref: https://arxiv.org/pdf/1506.01186.pdf
    """
    def __init__(self,
                 optimizer,
                 cyclical_steps,
                 min_lr=1e-3,
                 max_lr=1e-2):
        super().__init__(optimizer)
        assert cyclical_steps > 2
        assert max_lr >= min_lr
        self._cyclical_steps = cyclical_steps
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._abs_lr_delta = 2 * (max_lr - min_lr) / cyclical_steps

        self._is_cycling = False
        self._cycling_start_t = None

    def _compute_lr(self):
        pass
