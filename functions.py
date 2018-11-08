# coding: utf-8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_back(z, dz):
    return dz*sigmoid(z) * (1 - sigmoid(z))


def tanh(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y


def tanh_grad(x):
    y = x*(1-x**2)
    return y
