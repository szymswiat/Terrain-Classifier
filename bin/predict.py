import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model

from bin.train import IniConfig
from eval.eval_helpers import map_to_classes, iou_score, f1_score
from training.data_provider import ValBatchGenerator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for terrain predictor.')

    parser.add_argument('-i', '--images-dir', help='Path to dir with images.')
    parser.add_argument('-o', '--output-dir', help='Path to dir for predicted masks.')

    parser.add_argument('-m', '--model', help='Path to model.')

    return parser.parse_args(args)


def postprocess_data(images_batch, predictions, gt_masks_batch):
    b, h, w, ch = images_batch.shape
    gt_classes_map = map_to_classes(gt_masks_batch)
    gt_mask_to_display = np.zeros(shape=gt_classes_map.shape, dtype=np.uint8)
    gt_mask_to_display[gt_classes_map == 1] = 255
    gt_mask_to_display = gt_mask_to_display.reshape((b, h, w))

    pred_classes_map = map_to_classes(predictions)
    pred_mask_to_display = np.zeros(shape=pred_classes_map.shape, dtype=np.uint8)
    pred_mask_to_display[pred_classes_map == 1] = 255
    pred_mask_to_display = pred_mask_to_display.reshape((b, h, w))

    images_batch += 1.
    images_batch *= 127.5
    image = images_batch[0].astype(dtype=np.uint8)

    return image, pred_mask_to_display[0], gt_mask_to_display[0]


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

    dependencies = {
        'iou_score': iou_score,
        'f1_score': f1_score
    }

    model: Model = load_model(args.model, custom_objects=dependencies)

    for images_batch, gt_masks_batch in data_gen:
        predictions = model.predict_on_batch(images_batch)

        predictions = predictions.numpy()

        image, prediction, gt = postprocess_data(images_batch, predictions, gt_masks_batch)

        fig = plt.figure(
            dpi=200
        )
        gs = fig.add_gridspec(1, 3)

        fig.add_subplot(gs[0, 0])
        plt.imshow(image)
        fig.add_subplot(gs[0, 1])
        plt.imshow(prediction, cmap='gray')
        fig.add_subplot(gs[0, 2])
        plt.imshow(gt, cmap='gray')

        plt.show()


if __name__ == "__main__":
    main()
