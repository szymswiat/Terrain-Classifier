import numpy as np


def map_to_classes(predictions: np.ndarray):
    """
    Returns integer matrix with class numbers.
    """
    class_matrix = np.argmax(predictions, axis=-1)

    return class_matrix
