import os

from keras.utils import Sequence


class TrainBatchGenerator(Sequence):

    def __init__(self, data_path, steps, batch_size=1):
        super(TrainBatchGenerator, self).__init__()

        self.batch_size = batch_size
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        imgs_batch = []  # array dims - batch_size, height, width, channels
        gt_masks_batch = []

        #
        #
        # TODO generate training data batch
        #
        #

        raise NotImplemented()

        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        raise NotImplemented()


class ValBatchGenerator(Sequence):

    def __init__(self, data_path, generate_gt=True, batch_size=1):
        super(ValBatchGenerator, self).__init__()

        self.generate_gt = generate_gt

        self.img_names = os.listdir('{}/{}'.format(data_path, 'imgs'))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        imgs_batch = []  # array dims - batch_size, height, width, channels
        gt_masks_batch = []

        if self.generate_gt:
            # TODO zwrócić też maski
            pass

        #
        #
        # TODO generate val data batch
        #
        #

        raise NotImplemented()

        return imgs_batch, gt_masks_batch

    def on_epoch_end(self):
        super().on_epoch_end()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            self.n += 1
            return self[self.n]
        else:
            raise StopIteration()
