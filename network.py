from random import random, randrange
from activations import sigmoid, derivate


class Neuron:
    def __init__(self, prev_layer_length):
        # Generate a random list of weights
        # based on the length of the previous connecting layer
        self.weights = [random() for i in range(prev_layer_length)]
        self.output = 0.0
        self.delta = 0.0
        # TODO CHANGE BIAS
        self.bias = randrange(11)


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


def forward_propagate(network, initial_inputs):
    # Initialize inputs as the initial_inputs
    inputs = initial_inputs
    for layer in network.layers:
        # Generate the inputs for each layer
        new_inputs = []
        for neuron in layer:
            # Activate each neuron in the layer with the current inputs
            output = activate(neuron.weights, inputs, neuron.bias)
            # Run activation output through step function
            neuron_output = sigmoid(output)
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
        if l == len(network.layers) - 1:
            # If the current layer is the output layer
            # Compute the error by expected - output
            for i in range(len(layer.neurons)):
                errors.append(expected[i] - layer.neurons[i].output)
        else:
            # If the current layer is not the output layer
            # Compute the error
            for i in range(len(layer.neurons)):
                # For the neurons in the current layer
                # Compute error by summing the weight * delta of the neurons in the next layer
                error = 0.0
                for neuron in network.layers[l + 1]:
                    error += neuron.weights[i] * neuron.delta
                errors.append(error)
        for i in range(len(layer.neurons)):
            # Compute the delta of the neuron by multiplying the erro by the output derivate
            layer.neurons[i].delta = errors[i] * \
                derivate(layer.neurons[i].output)


def update_weights(network, initial_inputs, learn_rate):
    inputs = initial_inputs
    for i in range(len(network.layers)):
        if i != 0:
            # If the current layer is not the first layer
            # Get the inputs as the previous layers outputs
            inputs = []
            for neuron in network.layers[i - 1]:
                inputs.append(neuron.output)
        # For the neurons in the current layer
        for j in range(len(network.layers[i])):
            for k in range(len(inputs)):
                # Update the weights of the neurons in the current layer
                # With the product of the learning_rate, current neurons delta, and connecting input
                network.layers[i].neurons[j].weights[k] += learn_rate * \
                    network.layers[i].neurons[j].delta * inputs[k]


def sum_square_errors(expected, output):
    result = 0.0
    for i in range(len(expected)):
        result += (expected[i] - output[i])**2
    return result


def train(network, train_data, train_labels, learn_rate, epochs):
    for epoch in epochs:
        sum_error = 0
        for i in range(len(train_data)):
            outputs = forward_propagate(network, train_data[i])
            # Set the expected list to 0s using the length of the last layer
            expected = [0]*network.layers[-1]
            # Set expected labels index to 1
            expected[train_labels[i]] = 1.0
            sum_error += sum_square_errors(expected, outputs)
            backward_propagate(network, expected)
            update_weights(network, train_data[i], learn_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' %
              (epoch, learn_rate, sum_error))
