import argparse
import sys

import numpy as np
from tensorflow.keras.models import load_model, Model

from eval.eval_helpers import map_to_classes
from training.data_provider import ValBatchGenerator

from imageio.plugins.pillow import ndarray_to_pil


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for terrain predictor.')

    parser.add_argument('-i', '--images-dir', help='Path to dir with images.')
    parser.add_argument('-o', '--output-dir', help='Path to dir for predicted masks.')

    parser.add_argument('-m', '--model', help='Path to model.')

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    data_gen = ValBatchGenerator(
        data_path='./data/val',
        train_gen=None,
        batch_size=1,
        validation_images_size=0
    )

    model: Model = load_model(args.model)

    for images_batch, masks_batch in data_gen:
        predictions = model.predict_on_batch(images_batch)

        b, h, w, ch = images_batch.shape

        predictions = predictions.numpy()

        classes_output = map_to_classes(predictions)

        mask_to_display = np.zeros(shape=classes_output.shape, dtype=np.uint8)

        mask_to_display[classes_output == 1] = 255
        mask_to_display = mask_to_display.reshape((b, h, w))

        ndarray_to_pil(images_batch[0]).show()
        ndarray_to_pil(mask_to_display[0]).show()


if __name__ == "__main__":
    main()
