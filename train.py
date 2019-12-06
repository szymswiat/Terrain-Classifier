import configparser
import os

from keras.models import load_model
from keras.utils.vis_utils import plot_model as plot

# ========= Load settings from Config file
from model.model import get_unet
from training.data_provider import get_batch_gen

config = configparser.RawConfigParser()
config.read('configuration.ini')

# Training settings
n_epochs = int(config.get('training', 'n_epochs'))
# batch_size = int(config.get('training settings', 'batch_size'))
mode = config.get('training', 'mode')

batch_gen = get_batch_gen(
    imgs_path=config.get('data paths', 'imgs'),
    gt_masks_path=config.get('data paths', 'gt_masks')
)

# =========== Construct and or load the model architecture
model = None

model_output_dir = config.get('training', 'output')
model_name = config.get('model', 'name')

model_output_dir_path = '{}/{}'.format(model_output_dir, model_name)
if os.path.isdir(model_output_dir_path) is False:
    os.mkdir(model_output_dir_path)

if mode == 'new':
    n_ch = int(config.get('model', 'img_channels'))
    height = int(config.get('model', 'img_height'))
    width = int(config.get('model', 'img_width'))
    model = get_unet(n_ch, height, width)  # the U-net model
    print("Check: final output of the network:")
    print(model.output_shape)
    plot(model, to_file='{}/{}_model.png'.format(model_output_dir_path, model_name))  # check how the model looks like
    json_string = model.to_json()
    open('{}/{}_architecture.json'.format(model_output_dir_path, model_name), 'w').write(json_string)
elif mode == 'resume':
    model = load_model('{}/{}_whole_model.h5'.format(model_output_dir_path, model_name))

# ============  Training
model.fit_generator(generator=batch_gen, epochs=n_epochs, verbose=2)

# ========== Save model
model.save_weights('{}/{}_last_weights.h5'.format(model_output_dir_path, model_name), overwrite=True)
model.save('{}/{}_whole_model.h5'.format(model_output_dir_path, model_name))

print("\n\n========  Finished Training =======================")
