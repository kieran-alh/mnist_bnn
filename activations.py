from math import exp


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def derivate(output):
    return output * (1.0 - output)
