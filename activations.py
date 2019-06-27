from math import exp


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def sigmoid_derivate(output):
    return output * (1.0 - output)


def tanh_derivate(output):
    return 1 - (output**2)
