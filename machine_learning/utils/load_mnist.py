import gzip
import os
import struct
import numpy as np

__all__ = ['load_mnist']

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/mnist'))


def load_mnist(name='train'):
    if name == 'train':
        images_name = 'train-images-idx3-ubyte.gz'
        labels_name = 'train-labels-idx1-ubyte.gz'
    else:
        images_name = 't10k-images-idx3-ubyte.gz'
        labels_name = 't10k-labels-idx1-ubyte.gz'

    with gzip.open(os.path.join(path, images_name)) as f:
        head = f.read(16)
        data = f.read()
    _, n_images, n_rows, n_columns = struct.unpack('>IIII', head)
    x = np.frombuffer(data, dtype=np.uint8)

    with gzip.open(os.path.join(path, labels_name)) as f:
        head = f.read(8)
        data = f.read()
    _, n_labels = struct.unpack('>II', head)
    y = np.frombuffer(data, dtype=np.uint8)

    assert n_images == n_labels
    assert x.shape[0] == n_images * n_rows * n_columns
    assert y.shape[0] == n_images

    return x.reshape(n_images, n_rows * n_columns) / 255, y.astype(np.int64)
