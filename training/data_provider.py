def get_batch_gen(imgs_path, gt_masks_path):
    while True:
        imgs_batch = []  # array dims - channels, height, width - values 0 - 255 float
        gt_masks_batch = []  # array dims - channels, height, width - values 0 - 1 float

        #
        #
        # TODO generate training data batch
        #
        #

        raise NotImplemented()

        yield imgs_batch, gt_masks_batch
        pass
