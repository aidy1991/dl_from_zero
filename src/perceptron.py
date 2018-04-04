# -*- coding: utf-8 -*-
""" comment """

import numpy as np


def perceptron(x, w, b):
    s = np.sum(x * w) + b
    return 1 if s > 0 else 0


def AND(x):
    return perceptron(x, np.array([0.5, 0.5]), -0.7)


def OR(x):
    return perceptron(x, np.array([0.5, 0.5]), -0.4)


def NAND(x):
    return 0 if AND(x) else 1


def XOR(x):
    return AND(np.array([NAND(x), OR(x)]))


if __name__ == '__main__':
    print('AND')
    print('[0, 0]: ', AND(np.array([0, 0])))
    print('[0, 1]: ', AND(np.array([0, 1])))
    print('[1, 0]: ', AND(np.array([1, 0])))
    print('[1, 1]: ', AND(np.array([1, 1])))

    print('OR')
    print('[0, 0]: ', OR(np.array([0, 0])))
    print('[0, 1]: ', OR(np.array([0, 1])))
    print('[1, 0]: ', OR(np.array([1, 0])))
    print('[1, 1]: ', OR(np.array([1, 1])))

    print('NAND')
    print('[0, 0]: ', NAND(np.array([0, 0])))
    print('[0, 1]: ', NAND(np.array([0, 1])))
    print('[1, 0]: ', NAND(np.array([1, 0])))
    print('[1, 1]: ', NAND(np.array([1, 1])))

    print('XOR')
    print('[0, 0]: ', XOR(np.array([0, 0])))
    print('[0, 1]: ', XOR(np.array([0, 1])))
    print('[1, 0]: ', XOR(np.array([1, 0])))
    print('[1, 1]: ', XOR(np.array([1, 1])))
