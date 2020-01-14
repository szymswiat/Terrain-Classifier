import numpy as np
import tensorflow.keras.backend as K


def map_to_classes(predictions: np.ndarray):
    """
    Returns integer matrix with class numbers.
    """
    class_matrix = np.argmax(predictions, axis=-1)

    return class_matrix


def iou_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def f1_score(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice
