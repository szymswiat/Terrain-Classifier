import os

# ========= Load settings from Config file
from IniConfig import IniConfig
from callbacks.EvalCallback import EvalCallback
from eval.eval_helpers import iou_score, f1_score
from model.model import get_unet
from training.data_provider import TrainBatchGenerator, ValBatchGenerator
from tensorflow.keras.optimizers import Adam

debug = True


def create_callbacks(model_output_path, val_gen, c: IniConfig):
    callbacks = []

    eval = EvalCallback(val_gen, model_output_path, c)

    callbacks.append(eval)

    return callbacks


def main():
    c = IniConfig('configuration.ini')

    train_gen = TrainBatchGenerator(c)
    val_gen = ValBatchGenerator(c, train_gen)

    train_gen[0]

    model_output_dir_path = '{}/{}'.format(c.model_output_dir, c.model_name)
    if os.path.isdir(model_output_dir_path) is False:
        os.makedirs(model_output_dir_path)

    if c.mode == 'new':

        model = get_unet(c.n_classes, c.n_ch, c.img_size, c.img_size)  # the U-net model
        print(model.summary())
        if debug:
            from tensorflow.keras.utils import plot_model as plot
            plot(model, to_file='{}/{}_model.png'.format(model_output_dir_path, c.model_name))

        optimizer = Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[iou_score, f1_score]
        )
    elif c.mode == 'resume':

        dependencies = {
            'iou_score': iou_score,
            'f1_score': f1_score
        }

        # TODO select best model
        raise NotImplemented()
        # model = load_model('{}/{}_whole_model.h5'.format(model_output_dir_path, c.model_name))
    else:
        raise Exception("Invalid mode.")

    callbacks = create_callbacks(model_output_dir_path, val_gen, c)

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
