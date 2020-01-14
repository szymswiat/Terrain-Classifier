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


def clip_arrays_to_final_size(terrain, masks, req_size):
    masks_height, mask_width, n_classes = masks.shape

    clip_y1 = int((masks_height - req_size) / 2)
    clip_y2 = clip_y1 + req_size

    clip_x1 = int((mask_width - req_size) / 2)
    clip_x2 = clip_x1 + req_size

    clipped_terrain = terrain[clip_y1:clip_y2, clip_x1:clip_x2, :]
    clipped_forests = masks[clip_y1:clip_y2, clip_x1:clip_x2]
    return clipped_terrain, clipped_forests


def preprocess_pixels(image):
    image /= 127.5
    image -= 1.
    return image


class TrainBatchGenerator(Sequence):

    def __init__(self, config: IniConfig):
        super(TrainBatchGenerator, self).__init__()
        self.config = config

        self.data = {}
        self.classes = []

        self.batch_size = config.batch_size
        self.steps = config.train_steps
        self.img_max_rotation_angle = config.rotation_angle_range
        self.pre_rot_size = int(config.img_size * sqrt(2))

        self.load_data()

    """
        Loads data into self.data
    """

    def load_data(self):
        terrains_path = '{}/terrain'.format(self.config.train_data)
        terrain_names = os.listdir(terrains_path)

        assert len(terrain_names) is not 0

        self.classes = []

        with open(self.config.classes_filepath) as classes_file:
            self.classes.append(classes_file.readline().strip())
        assert len(self.classes) is not 0

        for terrain_name in terrain_names:
            single_terrain = {}
            masks_path = '{}/masks/{}'.format(self.config.train_data, terrain_name)

            terrain = Image.open('{}/{}'.format(terrains_path, terrain_name))
            single_terrain['terrain'] = np.array(terrain, dtype=np.float32)
            del terrain
            single_terrain['masks'] = {}

            for class_name in self.classes:
                mask = Image.open('{}/{}'.format(masks_path, class_name))
                single_terrain['masks'][class_name] = np.array(mask, dtype=np.uint8)
                del mask
            self.data[terrain_name] = single_terrain

        assert len(self.classes) + 1 == self.config.n_classes

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        img_size = self.config.img_size

        imgs_batch = np.zeros(shape=(self.batch_size, img_size, img_size, self.config.n_ch), dtype=np.float32)
        gt_masks_batch = np.zeros(shape=(self.batch_size, img_size * img_size, self.config.n_classes), dtype=np.float32)

        for x in range(self.batch_size):
            terrain_name, single_terrain = random.choice(list(self.data.items()))

            terrain_raster = single_terrain['terrain']

            rot_angle = random.randint(-self.img_max_rotation_angle, self.img_max_rotation_angle)

            terrain_slice, mask_volume = self.get_random_slice(terrain_raster, single_terrain['masks'])

            terrain_slice = imutils.rotate_bound(terrain_slice, rot_angle)
            mask_volume = imutils.rotate_bound(mask_volume, rot_angle)

            terrain_slice, mask_volume = clip_arrays_to_final_size(terrain_slice, mask_volume, img_size)

            mask_volume = mask_volume.reshape(mask_volume.shape[0] * mask_volume.shape[1], mask_volume.shape[2])

            bg_mask = np.zeros(mask_volume.shape[0])

            for i in range(len(self.classes)):
                mask_true_map = mask_volume[:, i] >= 100
                gt_masks_batch[x, mask_true_map, i + 1] = 1.0  # class mark
                bg_mask = np.logical_or(bg_mask, mask_true_map)

            gt_masks_batch[x, np.logical_not(bg_mask), 0] = 1.0  # class mark

            imgs_batch[x] = preprocess_pixels(terrain_slice)

        return imgs_batch, gt_masks_batch

    def get_random_slice(self, terrain, masks_dict):
        height, width, n_channels = terrain.shape

        pre_rot_size = self.pre_rot_size
        offset_y = random.randint(0, height - pre_rot_size)
        offset_y2 = offset_y + pre_rot_size
        offset_x = random.randint(0, width - pre_rot_size)
        offset_x2 = offset_x + pre_rot_size
        # TODO zrobić dla każdej klasy

        clipped_terrain = terrain[offset_y:offset_y2, offset_x:offset_x2, :]

        masks_volume = np.zeros(shape=(pre_rot_size, pre_rot_size, len(self.classes) + 1))  # +1 for shape compatibility
        for i, class_name in enumerate(self.classes):
            clipped_mask = masks_dict[class_name][offset_y:offset_y2, offset_x:offset_x2]
            masks_volume[:, :, i] = clipped_mask

        return clipped_terrain, masks_volume


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
