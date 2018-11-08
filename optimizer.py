# coding: utf-8
import numpy as np


def optimizer_SGD(lr, params, grads):
    for i in range(len(params)):
        params[i] -= lr * grads[i]

    return params
