import os
import random

import cv2
import h5py
import imutils as imutils
from keras.utils import Sequence
import numpy

from PIL import Image

Image.MAX_IMAGE_PIXELS = 128000000


def rotate_arrays_angle(terrain, forests, angle):
    rotated_terrain = imutils.rotate_bound(terrain, angle)
    rotated_forests = imutils.rotate_bound(forests, angle)
    return rotated_terrain, rotated_forests


def clip_arrays_to_final_size(terrain, forests):
    forests_x, forests_y = forests.shape

    clip_x1 = int((forests_x - 500) / 2)
    clip_x2 = clip_x1 + 500

    clip_y1 = int((forests_y - 500) / 2)
    clip_y2 = clip_y1 + 500

    clipped_terrain = terrain[clip_x1:clip_x2, clip_y1:clip_y2, 0:3]
    clipped_forests = forests[clip_x1:clip_x2, clip_y1:clip_y2]
    return clipped_terrain, clipped_forests


class TrainBatchGenerator(Sequence):
    img_size = 710
    img_max_rotation_angle = 45

    def __init__(self, data_path, steps, batch_size=1):
        super(TrainBatchGenerator, self).__init__()
        self.terrain = Image.open(data_path + '/CleanTerrain.tif')
        self.forests = Image.open(data_path + '/ForestsOnMapBits.tif')
        self.terrain_np = numpy.array(self.terrain)
        self.forests_np = numpy.array(self.forests)
        self.size_x, self.size_y, self.size_z = self.terrain_np.shape
        self.batch_size = batch_size
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        imgs_batch = []  # array dims - batch_size, height, width, channels
        gt_masks_batch = []  # array dims - batch_size, classifiers_size, height, width
        # fn there are 2 classifiers: forests and bg
        for x in range(self.batch_size):
            offset_x = random.randint(0, self.size_x - self.img_size)
            offset_x2 = offset_x + self.img_size
            offset_y = random.randint(0, self.size_y - self.img_size)
            offset_y2 = offset_y + self.img_size
            clipped_terrain_np = self.terrain_np[offset_x:offset_x2, offset_y:offset_y2, 0:3]
            clipped_forests_np = self.forests_np[offset_x:offset_x2, offset_y:offset_y2]
            terrain_rotated, forests_rotated = rotate_arrays_angle(clipped_terrain_np, clipped_forests_np,
                                                                   random.randint(-self.img_max_rotation_angle,
                                                                                  self.img_max_rotation_angle))
            final_terrain, final_forests = clip_arrays_to_final_size(terrain_rotated, forests_rotated)
            imgs_batch.append(final_terrain)
            masks = []
            forests_mask = [[None for y in range(500)] for x in range(500)]
            bg_mask = []
            for i in range(len(forests_mask)):
                for j in range(len(forests_mask[i])):
                    if final_forests[i][j] == 255.0:
                        forests_mask[i][j] = [1.0, 0.0]
                    else:
                        forests_mask[i][j] = [0.0, 1.0]
            gt_masks_batch.append(forests_mask)
        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        raise NotImplemented()


class ValBatchGenerator(Sequence):
    validation_file_name = 'validation_data.h5'
    validation_images_size = 1000
    dataset_len = 0

    def __init__(self, data_path, generate_gt=True, batch_size=1):
        super(ValBatchGenerator, self).__init__()
        self.data_path = data_path
        if not os.path.isfile('{}/{}'.format(data_path, self.validation_file_name)):
            self.generate_validation_data(data_path, self.validation_file_name, self.validation_images_size)

        with h5py.File('{}/{}'.format(data_path, self.validation_file_name), 'r') as f:
            data_set = f['images']
            self.dataset_len = int(len(data_set) / batch_size)

        self.batch_size = batch_size

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        imgs_batch = []  # array dims - batch_size, height, width, channels
        gt_masks_batch = []

        with h5py.File('{}/{}'.format(self.data_path, self.validation_file_name), 'r') as f:
            images = f['images']
            masks = f['masks']
            for i in range(self.batch_size):
                imgs_batch.append(images[index * self.batch_size + i])
                gt_masks_batch.append(masks[index * self.batch_size + i])

        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        self.n = 0
        return self

    def generate_validation_data(self, data_path, filename, validation_images_size):
        with h5py.File('{}/{}'.format(data_path, filename), 'w') as f:
            train_gen = TrainBatchGenerator(
                data_path=data_path,
                steps=0,
                batch_size=1
            )

            images = f.create_dataset('images', (validation_images_size, 500, 500, 3), dtype="uint8")
            masks = f.create_dataset('masks', (validation_images_size, 500, 500, 2), dtype="uint8")
            for i in range(validation_images_size):
                image, forest_mask = train_gen.__getitem__(0)
                images[i] = image[0]
                masks[i] = forest_mask[0]

    def __next__(self):
        if self.n < len(self):
            self.n += 1
            return self[self.n]
        else:
            raise StopIteration()
