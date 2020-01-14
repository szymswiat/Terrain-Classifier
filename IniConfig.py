import configparser


class IniConfig:
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
        self.img_size = int(parser.get('model', 'img_size'))

        self.train_data = parser.get('data paths', 'train_data')

        self.val_data = parser.get('data paths', 'val_data')

        self.train_steps = int(parser.get('training', 'train_steps'))

        self.rotation_angle_range = int(parser.get('training', 'rotation_angle_range'))

        self.log_filepath = parser.get('training', 'log_file')

        self.validation_images_count = int(parser.get('training', 'validation_images_count'))
