from random import random, randrange, gauss, seed
from math import tanh
from activations import sigmoid, sigmoid_derivative, tanhx, tanh_derivative, lrelu, lrelu_derivative

seed(1)


class Neuron:
    def __init__(self, prev_layer_length):
        # Generate a random list of weights
        # based on the length of the previous connecting layer
        self.weights = [gauss(0, 1) for i in range(prev_layer_length)]
        self.output = 0.0
        self.delta = 0.0
        self.bias = gauss(0, 1)

    def __str__(self):
        return 'O:{0}\nD:{1}\nB:{2}\nW:{3}'.format(self.output, self.delta, self.bias, self.weights)


class Layer:
    def __init__(self, prev_layer_length, cur_layer_length):
        self.neurons = [Neuron(prev_layer_length)
                        for i in range(cur_layer_length)]


class Network:
    def __init__(self, layer_list):
        self.layers = [Layer(layer_list[i], layer_list[i+1])
                       for i in range(len(layer_list) - 1)]


def activate(weights, inputs, bias):
    # Initialize the result as the bias
    result = bias
    # Run through all the weights (connected to this neuron)
    # and their associated inputs and add their product to the result
    for i in range(len(weights)):
        result += inputs[i] * weights[i]
    return result


def activation(value, func='tanh'):
    if func == 'tanh':
        return tanh(value)
    elif func == 'lrelu':
        return lrelu(value)
    else:
        return sigmoid(value)


def derivative(output, func='tanh'):
    if func == 'tanh':
        return tanh_derivative(output)
    elif func == 'lrelu':
        return lrelu_derivative(output)
    else:
        return sigmoid_derivative(output)


def forward_propagate(network, initial_inputs, read_only=False):
    # Initialize inputs as the initial_inputs
    inputs = initial_inputs
    for layer in network.layers:
        # Generate the inputs for each layer
        new_inputs = []
        for neuron in layer.neurons:
            # Activate each neuron in the layer with the current inputs
            output = activate(neuron.weights, inputs, neuron.bias)
            # Run activation output through step function
            neuron_output = activation(output, func='sigmoid')
            if not read_only:
                # Assign output to neuron
                neuron.output = neuron_output
            # Append the neuron's output to the list of new inputs
            new_inputs.append(neuron_output)
        # Update the inputs list with the newly generated inputs
        inputs = new_inputs
    # Return the final inputs (the inputs of the output layer)
    return inputs


def backward_propagate(network, expected):
    # Loop through the layers in the network
    # Beginning with the last layer
    for l in reversed(range(len(network.layers))):
        layer = network.layers[l]
        errors = []
        # For all the neurons in the current layer
        for i in range(len(layer.neurons)):
            if l == len(network.layers) - 1:
                # If the current layer is the output layer
                # Error is just expected - output
                errors.append(expected[i] - layer.neurons[i].output)
            else:
                # For all other layers
                # The neuron's error is the sum of all the weights * deltas that are connected
                # To the current neuron in the next layer
                error = 0.0
                for neuron in network.layers[l+1].neurons:
                    error += neuron.weights[i] * neuron.delta
                errors.append(error)
        for i in range(len(layer.neurons)):
            # Compute the delta of the neuron by multiplying the error by the output derivative
            layer.neurons[i].delta = errors[i] * derivative(layer.neurons[i].output, func='sigmoid')


def update_weights(network, initial_inputs, learn_rate):
    inputs = initial_inputs
    for i in range(len(network.layers)):
        if i != 0:
            # If the current layer is not the first layer
            # Get the inputs as the previous layers outputs
            inputs = []
            for neuron in network.layers[i - 1].neurons:
                inputs.append(neuron.output)
        # For the neurons in the current layer
        for j in range(len(network.layers[i].neurons)):
            for k in range(len(inputs)):
                # Update the weights of the neurons in the current layer
                # With the product of the learning_rate, current neurons delta, and connecting input
                weight_delta = learn_rate * \
                    network.layers[i].neurons[j].delta * inputs[k]
                network.layers[i].neurons[j].weights[k] += weight_delta
            # Update the neurons bias
            network.layers[i].neurons[j].bias += learn_rate * \
                network.layers[i].neurons[j].delta


def sum_square_errors(expected, output):
    result = 0.0
    for i in range(len(expected)):
        result += (expected[i] - output[i])**2
    return result


def print_neuron(network, l, n):
    print('LAYER %d' % l)
    print('NEURON %d' % n)
    print(max(network.layers[l].neurons[n].weights))
    print(network.layers[l].neurons[n])


def train(network, train_data, train_labels, learn_rate, epochs):
    for epoch in range(epochs):
        sum_error = 0
        for i in range(len(train_data)):
            outputs = forward_propagate(network, train_data[i])
            # Set the expected list to 0s using the length of the last layer
            expected = [0.0]*len(network.layers[-1].neurons)
            # Set expected labels index to 1
            expected[train_labels[i]] = 1.0
            sum_error += sum_square_errors(expected, outputs)
            backward_propagate(network, expected)
            update_weights(network, train_data[i], learn_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' %
              (epoch, learn_rate, sum_error))


def classify_network(network, inputs):
    return forward_propagate(network, inputs, True)


def single_network_output(outputs, threshold=0.5):
    value = outputs.index(max(outputs))
    if max(outputs) >= threshold:
        return value
    else:
        return -1


def calculate_network_accuracy(output, expected):
    value = 0
    for i in range(len(expected)):
        if output[i] == expected[i]:
            value += 1
    print('Amount Correct %d' % value)
    print('Amount Attempted %d' % len(output))
    return value / len(output)
