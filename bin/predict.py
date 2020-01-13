import argparse
import sys

import numpy as np
from imageio.plugins.pillow import ndarray_to_pil
from tensorflow.keras.models import load_model, Model

from bin.train import IniConfig
from eval.eval_helpers import map_to_classes
from training.data_provider import ValBatchGenerator


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

    c = IniConfig('configuration.ini')

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

        gt_classes_map = map_to_classes(masks_batch)
        gt_mask_to_display = np.zeros(shape=gt_classes_map.shape, dtype=np.uint8)
        gt_mask_to_display[gt_classes_map == 1] = 255
        gt_mask_to_display = gt_mask_to_display.reshape((b, h, w))

        pred_classes_map = map_to_classes(predictions)
        pred_mask_to_display = np.zeros(shape=pred_classes_map.shape, dtype=np.uint8)
        pred_mask_to_display[pred_classes_map == 1] = 255
        pred_mask_to_display = pred_mask_to_display.reshape((b, h, w))

        ndarray_to_pil(images_batch[0]).show(title='image')
        ndarray_to_pil(pred_mask_to_display[0]).show(title='prediction')
        ndarray_to_pil(gt_mask_to_display[0]).show(title='gt')


if __name__ == "__main__":
    main()
