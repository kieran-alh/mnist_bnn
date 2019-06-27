from mnist import read_images, read_labels, normalize_images
from network import Network, train, classify_network, single_network_output, calculate_network_accuracy
import os


path = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGE_FILE = '{0}/data/train-images-idx3-ubyte'.format(path)
TRAIN_LABEL_FILE = '{0}/data/train-labels-idx1-ubyte'.format(path)
TEST_IMAGE_FILE = '{0}/data/t10k-images-idx3-ubyte'.format(path)
TEST_LABEL_FILE = '{0}/data/t10k-labels-idx1-ubyte'.format(path)


def train_network():
    print('BEGIN_TRAINING')
    images = normalize_images(read_images(TRAIN_IMAGE_FILE))
    labels = read_labels(TRAIN_LABEL_FILE)
    network = Network([784, 16, 16, 10])
    train(network, images, labels, 0.5, 20)
    training_values = []
    for i in range(len(images)):
        output_value = single_network_output(
            classify_network(network, images[i]))
        training_values.append(output_value)
    accuracy = calculate_network_accuracy(training_values, labels)
    print('TRAIN_NETWORK')
    print(accuracy)
    return network


def test_network(network):
    print('BEGIN_TESTING')
    images = normalize_images(read_images(TEST_IMAGE_FILE))
    labels = read_labels(TEST_LABEL_FILE)
    training_values = []
    for i in range(len(images)):
        output_value = single_network_output(
            classify_network(network, images[i]))
        training_values.append(output_value)
    accuracy = calculate_network_accuracy(training_values, labels)
    print('TEST_NETWORK')
    print(accuracy)


def run_neural_network():
    network = train_network()
    test_network(network)


if __name__ == "__main__":
    run_neural_network()
