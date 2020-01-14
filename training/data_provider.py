import os
import random
from math import sqrt

import h5py
import imutils as imutils
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

Image.MAX_IMAGE_PIXELS = 128000000


def rotate_arrays_angle(terrain, forests, angle):
    rotated_terrain = imutils.rotate_bound(terrain, angle)
    rotated_forests = imutils.rotate_bound(forests, angle)
    return rotated_terrain, rotated_forests


def clip_arrays_to_final_size(terrain, forests, req_size):
    forests_y, forests_x = forests.shape

    req_size_y, req_size_x = req_size

    clip_y1 = int((forests_y - req_size_y) / 2)
    clip_y2 = clip_y1 + req_size_y

    clip_x1 = int((forests_x - req_size_x) / 2)
    clip_x2 = clip_x1 + req_size_x

    clipped_terrain = terrain[clip_y1:clip_y2, clip_x1:clip_x2, :]
    clipped_forests = forests[clip_y1:clip_y2, clip_x1:clip_x2]
    return clipped_terrain, clipped_forests


def preprocess_pixels(image):
    image /= 127.5
    image -= 1.
    return image


class TrainBatchGenerator(Sequence):

    def __init__(self, data_path, steps, size_data, rot_angle, batch_size=1):
        super(TrainBatchGenerator, self).__init__()
        terrain = Image.open('{}/CleanTerrain.tif'.format(data_path))
        forests = Image.open('{}/ForestsOnMapBits.tif'.format(data_path))
        self.terrain_raster = np.array(terrain, dtype=np.float32)
        self.forest_mask_raster = np.array(forests)
        size_y, size_x, n_ch = self.terrain_raster.shape
        self.terrain_raster_size = (size_y, size_x)
        self.batch_size = batch_size
        self.steps = steps
        self.size_data = size_data
        self.img_max_rotation_angle = rot_angle
        self.pre_rot_size = (int(size_data['height'] * sqrt(2)), int(size_data['width'] * sqrt(2)))

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        height, width = self.size_data['height'], self.size_data['width']

        imgs_batch = np.zeros(shape=(self.batch_size, height, width, self.size_data['channels']), dtype=np.float32)
        gt_masks_batch = np.zeros(shape=(self.batch_size, height * width, self.size_data['classes']), dtype=np.float32)
        # fn there are 2 classifiers: forests and bg
        terrain_raster_y, terrain_raster_x = self.terrain_raster_size
        for x in range(self.batch_size):
            pre_rot_height, pre_rot_width = self.pre_rot_size
            offset_y = random.randint(0, terrain_raster_y - pre_rot_height)
            offset_y2 = offset_y + pre_rot_height
            offset_x = random.randint(0, terrain_raster_x - pre_rot_width)
            offset_x2 = offset_x + pre_rot_width
            clipped_terrain_np = self.terrain_raster[offset_y:offset_y2, offset_x:offset_x2, :]
            clipped_forests_np = self.forest_mask_raster[offset_y:offset_y2, offset_x:offset_x2]
            terrain_rotated, forests_rotated = rotate_arrays_angle(clipped_terrain_np, clipped_forests_np,
                                                                   random.randint(-self.img_max_rotation_angle,
                                                                                  self.img_max_rotation_angle))
            final_terrain, final_forests = clip_arrays_to_final_size(terrain_rotated, forests_rotated, (height, width))
            imgs_batch[x] = preprocess_pixels(final_terrain)

            final_forests = final_forests.reshape(final_forests.shape[0] * final_forests.shape[1])
            gt_masks_batch[x, final_forests < 100, 0] = 1.0  # bg class
            gt_masks_batch[x, final_forests >= 100, 1] = 1.0  # 1st class

        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        raise NotImplemented()

    def generate_validation_data(self, h5filepath, validation_images_size):
        batch_size_cp = self.batch_size
        self.batch_size = 1
        height, width = self.size_data['height'], self.size_data['width']
        channels, classes = self.size_data['channels'], self.size_data['classes']
        with h5py.File(h5filepath, 'w') as f:
            images = f.create_dataset('images', (validation_images_size, height, width, channels), dtype="float32")
            masks = f.create_dataset('masks', (validation_images_size, height * width, classes), dtype="uint8")
            for i in range(validation_images_size):
                image, forest_mask = self[0]
                images[i] = image[0]
                masks[i] = forest_mask[0]
        self.batch_size = batch_size_cp


class ValBatchGenerator(Sequence):
    validation_file_name = 'validation_data.h5'
    dataset_len = 0

    def __init__(self, data_path, train_gen, validation_images_size, batch_size=1):
        super(ValBatchGenerator, self).__init__()
        self.data_path = data_path
        h5filepath = '{}/{}'.format(data_path, self.validation_file_name)
        if not os.path.isfile(h5filepath):
            print('\nValidation data does not exits. Generating ...\n')
            train_gen.generate_validation_data(h5filepath, validation_images_size)

        self.data_file = h5py.File(h5filepath, 'r')

        data_set = self.data_file['images']
        self.dataset_len = int(len(data_set) / batch_size)

        self.batch_size = batch_size
        self.train_gen = train_gen

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        images = self.data_file['images']
        masks = self.data_file['masks']
        start_slice = index * self.batch_size
        end_slice = index * self.batch_size + self.batch_size

        images = images[start_slice:end_slice].astype(dtype=np.float32)
        masks = masks[start_slice:end_slice].astype(dtype=np.float32)

        return images, masks

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            self.n += 1
            return self[self.n - 1]
        else:
            raise StopIteration()
