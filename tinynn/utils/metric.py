"""Methods to compute common machine learning metrics"""

import numpy as np


def _roc_curve(preds, targets, partition, pos_class, neg_class):
    """ROC curve (for binary classification only)"""
    pass


def auc_roc_curve(preds, targets, partition=300, pos_class=1, neg_class=0):
    """Area unser the ROC curve (for binary classification only)"""
    pass


def auc(preds, targets, pos_class=1, neg_class=0):
    pass


def accuracy(preds, targets):
    pass


def log_loss(preds, targets):
    pass


def precision(preds, targets, pos_class=1, neg_class=0):
    """precision = TP / (TP + FP)"""
    pass


def recall(preds, targets, pos_class=1, neg_class=0):
    """recall = TP / (TP + FN)"""
    pass


def f1_score(preds, targets, pos_class=1, neg_class=0):
    pass


def explained_variation(preds, targets):
    """
    Computes fraction of variance that pred_y explains about y.
    Returns 1 - Var[y-pred_y] / Var[y]

    Interpretation:
        EV=0  =>  might as well have predicted zero
        EV=1  =>  perfect prediction
        EV<0  =>  worse than just predicting zero
    """
    pass


def r_square(preds, targets):
    pass


def mean_square_error(preds, targets):
    pass


def mean_absolute_error(preds, targets):
    pass
