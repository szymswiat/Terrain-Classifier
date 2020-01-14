import os
import random
from math import sqrt

import h5py
import imutils as imutils
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

from IniConfig import IniConfig

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

    def __init__(self, config: IniConfig):
        super(TrainBatchGenerator, self).__init__()
        terrain = Image.open('{}/terrain/CleanTerrain.tif'.format(config.train_data))
        forests = Image.open('{}/masks/ForestsOnMapBits.tif'.format(config.train_data))

        self.config = config

        self.terrain_raster = np.array(terrain, dtype=np.float32)
        self.forest_mask_raster = np.array(forests)
        size_y, size_x, n_ch = self.terrain_raster.shape
        self.terrain_raster_size = (size_y, size_x)
        self.batch_size = config.batch_size
        self.steps = config.train_steps
        self.img_max_rotation_angle = config.rotation_angle_range
        self.pre_rot_size = int(config.img_size * sqrt(2))

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        img_size = self.config.img_size

        imgs_batch = np.zeros(shape=(self.batch_size, img_size, img_size, self.config.n_ch), dtype=np.float32)
        gt_masks_batch = np.zeros(shape=(self.batch_size, img_size * img_size, self.config.n_classes), dtype=np.float32)
        # fn there are 2 classifiers: forests and bg
        terrain_raster_y, terrain_raster_x = self.terrain_raster_size
        for x in range(self.batch_size):
            pre_rot_size = self.pre_rot_size
            offset_y = random.randint(0, terrain_raster_y - pre_rot_size)
            offset_y2 = offset_y + pre_rot_size
            offset_x = random.randint(0, terrain_raster_x - pre_rot_size)
            offset_x2 = offset_x + pre_rot_size
            clipped_terrain_np = self.terrain_raster[offset_y:offset_y2, offset_x:offset_x2, :]
            clipped_forests_np = self.forest_mask_raster[offset_y:offset_y2, offset_x:offset_x2]
            terrain_rotated, forests_rotated = rotate_arrays_angle(clipped_terrain_np, clipped_forests_np,
                                                                   random.randint(-self.img_max_rotation_angle,
                                                                                  self.img_max_rotation_angle))
            final_terrain, final_forests = clip_arrays_to_final_size(terrain_rotated, forests_rotated, (img_size, img_size))
            imgs_batch[x] = preprocess_pixels(final_terrain)

            final_forests = final_forests.reshape(final_forests.shape[0] * final_forests.shape[1])
            gt_masks_batch[x, final_forests < 100, 0] = 1.0  # bg class
            gt_masks_batch[x, final_forests >= 100, 1] = 1.0  # 1st class

        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        raise NotImplemented()


class ValBatchGenerator(Sequence):
    dataset_len = 0

    def __init__(self, config: IniConfig, train_gen=None):
        super(ValBatchGenerator, self).__init__()
        self.data_path = config.val_data
        self.batch_size = 1
        h5filepath = '{}/size-{}-count-{}.h5py'.format(self.data_path, config.img_size, config.validation_images_count)

        if not os.path.isfile(h5filepath):
            if train_gen is None:
                train_gen = TrainBatchGenerator(config)
            print('\nValidation data does not exist. Generating ...\n')
            generate_validation_data(train_gen, h5filepath, config.validation_images_count)

        self.data_file = h5py.File(h5filepath, 'r')

        data_set = self.data_file['images']
        self.dataset_len = int(len(data_set) / self.batch_size)

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


def generate_validation_data(train_gen: TrainBatchGenerator, h5filepath, validation_images_size):
    batch_size_cp = train_gen.batch_size
    train_gen.batch_size = 1
    img_size = train_gen.config.img_size
    channels, classes = train_gen.config.n_ch, train_gen.config.n_classes
    with h5py.File(h5filepath, 'w') as f:
        images = f.create_dataset('images', (validation_images_size, img_size, img_size, channels), dtype="float32")
        masks = f.create_dataset('masks', (validation_images_size, img_size * img_size, classes), dtype="uint8")
        for i in range(validation_images_size):
            image, mask = train_gen[0]
            images[i] = image[0]
            masks[i] = mask[0]
    train_gen.batch_size = batch_size_cp
