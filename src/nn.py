# -*- coding: utf-8 -*-
""" comment """

import numpy as np


def identity_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def softmax_function(x):
    # オーバーフローを防ぐために最大値を全体から引く
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = sum(exp_x)
    return exp_x / sum_exp_x


class Network:
    def __init__(self):
        self.W = np.array([
            [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            [[0.1, 0.3], [0.2, 0.4]],
        ])
        self.B = np.array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2],
            [0.1, 0.2],
        ])

    def forward(self, x):
        A1 = np.dot(x, self.W[0]) + self.B[0]
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, self.W[1]) + self.B[1]
        Z2 = sigmoid(A2)
        A3 = np.dot(Z2, self.W[2]) + self.B[2]
        Z3 = identity_function(A3)
        return Z3


if __name__ == '__main__':
    network = Network()
    x = np.array([1.0, 0.5])
    output = network.forward(x)
    print(output)
    print(softmax_function(np.array([0.3, 2.9, 4.0])))
