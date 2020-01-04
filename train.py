import configparser
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.vis_utils import plot_model as plot

# ========= Load settings from Config file
from model.model import get_unet
from training.data_provider import get_batch_gen


class Config:
    def __init__(self, filepath):
        parser = configparser.RawConfigParser()
        parser.read(filepath)

        self.n_epochs = int(parser.get('training', 'n_epochs'))

        self.mode = parser.get('training', 'mode')
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.model_output_dir = parser.get('training', 'output')
        self.model_name = parser.get('model', 'name')

        self.n_classes = int(parser.get('model', 'n_classes'))
        self.n_ch = int(parser.get('model', 'img_channels'))
        self.height = int(parser.get('model', 'img_height'))
        self.width = int(parser.get('model', 'img_width'))

        self.imgs_path_train = parser.get('data paths', 'imgs_train')
        self.gt_masks_path_train = parser.get('data paths', 'gt_masks_train')

        self.imgs_path_val = parser.get('data paths', 'imgs_val')
        self.gt_masks_path_val = parser.get('data paths', 'gt_masks_val')

        self.train_steps = int(parser.get('training', 'train_steps'))
        self.val_steps = int(parser.get('training', 'val_steps'))


def create_callbacks(weights_dir):
    callbacks = []

    # checkpoint
    filepath = '{}/{}'.format(weights_dir, "model-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callbacks.append(checkpoint)

    return callbacks


def main():
    c = Config('configuration.ini')

    train_gen = get_batch_gen(
        imgs_path=c.imgs_path_train,
        gt_masks_path=c.gt_masks_path_train
    )
    val_gen = get_batch_gen(
        imgs_path=c.imgs_path_val,
        gt_masks_path=c.gt_masks_path_val
    )

    # =========== Construct and or load the model architecture
    model = None

    model_output_dir_path = '{}/{}'.format(c.model_output_dir, c.model_name)
    if os.path.isdir(model_output_dir_path) is False:
        os.mkdir(model_output_dir_path)

    if c.mode == 'new':

        model = get_unet(c.n_classes, c.n_ch, c.height, c.width)  # the U-net model
        print("Check: final output of the network:")
        print(model.summary())
        plot(model,
             to_file='{}/{}_model.png'.format(model_output_dir_path, c.model_name))  # check how the model looks like
        json_string = model.to_json()
        open('{}/{}_architecture.json'.format(model_output_dir_path, c.model_name), 'w').write(json_string)

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    elif c.mode == 'resume':
        model = load_model('{}/{}_whole_model.h5'.format(model_output_dir_path, c.model_name))
    else:
        raise Exception("Invalid mode.")

    callbacks = create_callbacks(model_output_dir_path)

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=c.train_steps,
        validation_data=val_gen,
        validation_steps=c.val_steps,
        epochs=c.n_epochs,
        callbacks=callbacks,
        verbose=2)

    print("\n\n======================= Training finished =======================")


if __name__ == "__main__":
    main()
