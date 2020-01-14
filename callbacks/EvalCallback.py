from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

from IniConfig import IniConfig


def m_print(file, msg):
    print(msg)
    file.write(msg)
    file.flush()


class EvalCallback(Callback):

    def __init__(self, val_gen: Sequence, model_output_path, config: IniConfig):
        super(EvalCallback, self).__init__()

        self.val_gen = val_gen

        self.log_file = open('{}/{}'.format(model_output_path, config.log_filepath), 'w')
        self.model_output_path = model_output_path

        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        self.log_file.write('epoch {}\n'.format(epoch))

        m_print(self.log_file, '\ntrain:\n loss - {}\n iou - {}\n f1 - {}\n'.format(
            logs['loss'], logs['iou_score'], logs['f1_score']
        ))
        m_print(self.log_file, 'val:\n loss - {}\n iou - {}\n f1 - {}\n'.format(
            logs['val_loss'], logs['val_iou_score'], logs['val_f1_score']
        ))

        if logs['val_f1_score'] > self.best_val_f1:
            self.best_val_f1 = logs['val_f1_score']
            print('Better score achieved, saving model.')
            self.model.save('{}/epoch_{}.h5'.format(self.model_output_path, epoch))

    def on_train_end(self, logs=None):
        self.log_file.close()
