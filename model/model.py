from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Reshape
from tensorflow.keras.models import Model


def conv_block(filters, inputs):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = Dropout(0.2)(conv)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    return conv


def pooling_block(filters, inputs):
    conv = conv_block(filters, inputs)
    pool = MaxPooling2D((2, 2))(conv)

    return conv, pool


def up_block(filters, inputs):
    conv = conv_block(filters, inputs)
    up = UpSampling2D(size=(2, 2))(conv)

    return up


def get_unet(n_classes, n_ch, img_height, img_width) -> Model:
    inputs = Input(shape=(img_height, img_width, n_ch))

    level_1, pooled_level_1 = pooling_block(filters=32, inputs=inputs)
    level_2, pooled_level_2 = pooling_block(filters=64, inputs=pooled_level_1)
    level_3, pooled_level_3 = pooling_block(filters=128, inputs=pooled_level_2)
    level_4, pooled_level_4 = pooling_block(filters=256, inputs=pooled_level_3)

    up_level_4 = up_block(filters=512, inputs=pooled_level_4)
    concat_level_4 = concatenate([up_level_4, level_4], axis=3)

    up_level_3 = up_block(filters=256, inputs=concat_level_4)
    concat_level_3 = concatenate([up_level_3, level_3], axis=3)

    up_level_2 = up_block(filters=128, inputs=concat_level_3)
    concat_level_2 = concatenate([up_level_2, level_2], axis=3)

    up_level_1 = up_block(filters=64, inputs=concat_level_2)
    concat_level_1 = concatenate([up_level_1, level_1], axis=3)

    level_1_output = conv_block(filters=32, inputs=concat_level_1)

    output = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(level_1_output)
    shaped_output = Reshape((img_height * img_width, n_classes))(output)
    # conv6 = core.Permute((2, 1))(conv6)

    conv7 = Activation('softmax')(shaped_output)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)

    return model
