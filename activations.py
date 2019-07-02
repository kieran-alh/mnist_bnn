from math import exp


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def sigmoid_derivate(output):
    return output * (1.0 - output)


def tanhx(activation):
    num = exp(activation) - exp(-activation)
    den = exp(activation) + exp(-activation)
    return num / den


def tanh_derivate(output):
    return 1 - (output**2)
