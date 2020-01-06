import argparse
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model

from training.data_provider import ValBatchGenerator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for terrain predictor.')

    parser.add_argument('-i', '--images-dir', help='Path to dir with images.')
    parser.add_argument('-o', '--output-dir', help='Path to dir for predicted masks.')

    parser.add_argument('-m', '--model', help='Path to model.')

    return parser.parse_args(args)


def map_to_classes(predictions: np.ndarray):
    """
    Returns integer matrix with class numbers.
    """
    class_matrix = np.argmax(predictions, axis=4)

    return class_matrix


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    data_gen = ValBatchGenerator(
        data_path=args.images_dir
    )

    model: Model = load_model(args.model)

    for X, Y in data_gen:
        predictions = model.predict_on_batch(X)

        classes_output = map_to_classes(predictions)

        # TODO do sth with output


if __name__ == "__main__":
    main()
