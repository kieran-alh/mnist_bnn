def read_images(file_name):
    """
    Read Image binary file of MNIST dataset
    :param file_name: Path to images file
    :returns: List of image lists
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


def normalize_images(images):
    """
    Normalize image pixel value between [0.01, 1.0)
    :param file_name: List of raw image lists
    :returns: List of 2D image lists (normalized)
    """
    print('Normalizing')
    normalized = []
    for image in images:
        normalized.append(
            list(map(lambda x: x * (0.99 / 255.0) + 0.01, image)))
    print('Normalizing Complete')
    return normalized


def normalize_images_255(images):
    """
    Normalize image pixel value between [0.0, 1.0)
    :param file_name: List of raw image lists
    :returns: List of 2D image lists (normalized)
    """
    print('Normalizing')
    normalized = []
    for image in images:
        normalized.append(
            list(map(lambda x: x / 255.0, image)))
    print('Normalizing Complete')
    return normalized


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


def print_image(image, row, col):
    for i in range(row):
        for j in range(col):
            if image[(i*row) + j] > 0.01:
                print("*", end='')
            else:
                print(" ", end='')
        print()
