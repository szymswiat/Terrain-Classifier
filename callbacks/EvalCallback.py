from progressbar import progressbar
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

from eval.eval_helpers import map_to_classes


class EvalCallback(Callback):

    def __init__(self, val_gen: Sequence, size_data, model_output_path, log_filename):
        super(EvalCallback, self).__init__()

        self.val_gen = val_gen
        self.size_data = size_data

        self.log_file = open('{}/{}'.format(model_output_path, log_filename), 'w')
        self.model_output_path = model_output_path

    def on_epoch_end(self, epoch, logs=None):
        sd = self.size_data
        batch_size = self.val_gen.batch_size

        n_classes = sd['classes']

        self.log_file.write('epoch {}\n'.format(epoch))

        tp = 0
        for i in progressbar(range(len(self.val_gen)), prefix='Running network: '):
            imgs_batch, masks_batch = self.val_gen[i]

            predictions = self.model.predict_on_batch(imgs_batch).numpy()

            predictions = predictions.reshape(batch_size, sd['height'], sd['width'], n_classes)
            masks_batch = masks_batch.reshape(batch_size, sd['height'], sd['width'], n_classes)

            pred_map = map_to_classes(predictions)
            gt_map = map_to_classes(masks_batch)

            # TODO temporary for 2 classes

            tp += (pred_map == gt_map).sum()

        avg_tp = tp / (pred_map.size * len(self.val_gen))

        acc_string = 'avg - matches = {}%\n'.format(avg_tp * 100)
        print(acc_string)
        self.log_file.write(acc_string)

        self.model.save('{}/epoch_{}.h5'.format(self.model_output_path, epoch))

    def on_train_end(self, logs=None):
        self.log_file.close()
