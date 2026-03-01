"""MiniXception architecture for facial expression recognition."""

from tensorflow.keras import layers, Model, regularizers


NUM_CLASSES = 7


def residual_separable_block(x, num_filters, l2_reg=1e-4, dropout_rate=0.1):
    """Residual block with separable convolutions, BN, and optional dropout."""
    res_args = {"padding": "same", "use_bias": False}
    sep_args = {"padding": "same", "use_bias": False}
    if l2_reg > 0:
        res_args["kernel_regularizer"] = regularizers.l2(l2_reg)
        sep_args["depthwise_regularizer"] = regularizers.l2(l2_reg)
        sep_args["pointwise_regularizer"] = regularizers.l2(l2_reg)

    residual = layers.Conv2D(num_filters, (1, 1), strides=(2, 2), **res_args)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(num_filters, (3, 3), **sep_args)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(num_filters, (3, 3), **sep_args)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Add()([x, residual])
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_mini_xception(input_shape=(48, 48, 1), num_classes=NUM_CLASSES,
                        l2_reg=1e-4, dropout_rate=0.1):
    """Build the MiniXception model.

    Base module: 2x Conv2D(8) + 4 residual blocks [16, 32, 64, 128]
    + Conv2D(num_classes) -> GAP -> Softmax.
    """
    conv_args = {"padding": "same", "use_bias": False}
    final_args = {"padding": "same"}
    if l2_reg > 0:
        conv_args["kernel_regularizer"] = regularizers.l2(l2_reg)
        final_args["kernel_regularizer"] = regularizers.l2(l2_reg)

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(8, (3, 3), **conv_args)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, (3, 3), **conv_args)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for f in [16, 32, 64, 128]:
        x = residual_separable_block(x, f, l2_reg, dropout_rate)

    x = layers.Conv2D(num_classes, (3, 3), **final_args)(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation("softmax")(x)
    return Model(inputs=inp, outputs=out)
