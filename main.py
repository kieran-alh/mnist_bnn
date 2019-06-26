from mnist import read_images, read_labels, DATA_SIZE, ROWS, COLS
from network import Network, train
import os


path = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = '{0}/data/train-images-idx3-ubyte'.format(path)
LABEL_FILE = '{0}/data/train-labels-idx1-ubyte'.format(path)


def train_network():
    images = read_images(IMAGE_FILE)
    labels = read_labels(LABEL_FILE)
    network = Network([784, 16, 16, 10])
    train(network, images, labels, 0.25, 20)


if __name__ == "__main__":
    train_network()
