from math import exp, sinh, cosh


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def sigmoid_derivative(output):
    return output * (1.0 - output)


def tanhx(activation):
    return sinh(activation) / cosh(activation)


def tanh_derivative(output):
    return 1 - (output**2)


def lrelu(z, a=0.01):
    if z > 0:
        return z
    elif z <= 0:
        return z * a


def lrelu_derivative(z, a=0.01):
    if z > 0:
        return 1
    elif z <= 0:
        return a
