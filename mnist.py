DATA_SIZE = 60000
ROWS = 28
COLS = 28


def read_images(file_name):
    """
    Read Image binary file of MNIST dataset
    :param file_name: Path to images file
    :returns: List of 2D image lists (matrices)
    """
    print('Reading Images')
    file = open(file_name, 'rb')
    magic_number = int.from_bytes(file.read(4), 'big')
    total_images = int.from_bytes(file.read(4), 'big')
    row_size = int.from_bytes(file.read(4), 'big')
    col_size = int.from_bytes(file.read(4), 'big')
    image_size = row_size * col_size
    images = []
    for i in range(total_images):
        images.append(bytearray(file.read(image_size)))
    file.close()
    print('Images Complete')
    return images


def read_labels(file_name):
    """
    Read Label binary file of MNIST dataset
    :param file_name: Path to labels file
    :returns: List of int labels
    """
    print('Reading Labels')
    file = open(file_name, 'rb')
    magic_number = int.from_bytes(file.read(4), 'big')
    total_labels = int.from_bytes(file.read(4), 'big')
    labels = []
    for i in range(total_labels):
        labels.append(int.from_bytes(file.read(1), 'big'))
    file.close()
    print('Labels Complete')
    return labels


def print_image(image_matrix):
    for row in image_matrix:
        for col in row:
            if col > 1:
                print("*", end='')
            else:
                print(" ", end='')
        print()
