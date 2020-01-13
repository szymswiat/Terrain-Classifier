import configparser
import os

from keras.callbacks import ModelCheckpoint

# ========= Load settings from Config file
from model.model import get_unet
from training.data_provider import TrainBatchGenerator, ValBatchGenerator


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

        self.train_data = parser.get('data paths', 'train_data')

        self.val_data = parser.get('data paths', 'val_data')

        self.train_steps = int(parser.get('training', 'train_steps'))

        self.rotation_angle_range = int(parser.get('training', 'rotation_angle_range'))


def create_callbacks(weights_dir):
    callbacks = []

    # checkpoint
    filepath = '{}/{}'.format(weights_dir, "model-{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callbacks.append(checkpoint)

    return callbacks


def main():
    c = Config('configuration.ini')

    size_data = {
        'height': c.height,
        'width': c.width,
        'channels': c.n_ch,
        'classes': c.n_classes
    }

    train_gen = TrainBatchGenerator(
        data_path=c.train_data,
        steps=c.train_steps,
        batch_size=c.batch_size,
        size_data=size_data,
        rot_angle=c.rotation_angle_range
    )

    val_gen = ValBatchGenerator(
        data_path=c.val_data,
        train_gen=train_gen,
        batch_size=c.batch_size
    )

    model_output_dir_path = '{}/{}'.format(c.model_output_dir, c.model_name)
    if os.path.isdir(model_output_dir_path) is False:
        os.mkdir(model_output_dir_path)

    if c.mode == 'new':

        model = get_unet(c.n_classes, c.n_ch, c.height, c.width)  # the U-net model
        print(model.summary())

        model.compile(optimizer='sgd', loss='categorical_crossentropy')
    elif c.mode == 'resume':

        # TODO select best model
        raise NotImplemented()
        # model = load_model('{}/{}_whole_model.h5'.format(model_output_dir_path, c.model_name))
    else:
        raise Exception("Invalid mode.")

    callbacks = create_callbacks(model_output_dir_path)

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=c.n_epochs,
        callbacks=callbacks,
        verbose=1)


if __name__ == "__main__":
    main()
