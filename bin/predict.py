import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model

from IniConfig import IniConfig
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
    images_batch = images_batch[0]
    gt_masks_batch = gt_masks_batch[0]
    predictions = predictions[0]
    h, w, ch = images_batch.shape
    gt_classes_map = map_to_classes(gt_masks_batch)
    pred_classes_map = map_to_classes(predictions)

    gt_masks = []
    pred_masks = []

    for i in range(predictions.shape[-1]):
        gt_class_mask = np.zeros(shape=gt_classes_map.shape, dtype=np.uint8)
        gt_class_mask[gt_classes_map == i] = 255
        gt_class_mask = gt_class_mask.reshape((h, w))

        pred_class_mask = np.zeros(shape=pred_classes_map.shape, dtype=np.uint8)
        pred_class_mask[pred_classes_map == i] = 255
        pred_class_mask = pred_class_mask.reshape((h, w))

        gt_masks.append(gt_class_mask)
        pred_masks.append(pred_class_mask)

    images_batch += 1.
    images_batch *= 127.5
    image = images_batch.astype(dtype=np.uint8)

    return image, gt_masks, pred_masks


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    c = IniConfig('configuration.ini')

    data_gen = ValBatchGenerator(c)

    dependencies = {
        'iou_score': iou_score,
        'f1_score': f1_score
    }

    model: Model = load_model(args.model, custom_objects=dependencies)

    for images_batch, gt_masks_batch in data_gen:
        predictions = model.predict_on_batch(images_batch)

        predictions = predictions.numpy()

        image, gt_masks, pred_masks = postprocess_data(images_batch, predictions, gt_masks_batch)

        fig = plt.figure(
            dpi=200
        )
        gs = fig.add_gridspec(3, len(pred_masks))

        fig.add_subplot(gs[0, 1])
        plt.imshow(image)

        for i in range(len(pred_masks)):
            fig.add_subplot(gs[1, i])
            plt.imshow(pred_masks[i], cmap='gray')
            fig.add_subplot(gs[2, i])
            plt.imshow(gt_masks[i], cmap='gray')

        plt.show()


if __name__ == "__main__":
    main()
