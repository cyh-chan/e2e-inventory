from math import ceil, log
from tensorflow.keras.layers import Input, Lambda, Embedding, Conv1D, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import keras


def create_e2e_model(
        seq_len,
        n_dyn_fea=1,
        n_outputs=1,
        n_dilated_layers=3,
        kernel_size=2,
        n_filters=3,
        dropout_rate=0.1,
        max_cat_id=[100, 100],
):
    """Create E2E inventory replenishment model,
        with a Dilated CNN model for demand forecast.
        Demand forecast horizon is specified by n_outputs
        Future VLT is forecasted by 1 step ahead
        Optimal Reorder Quantity is predicted by 1 step ahead also

    Args:
        seq_len (int): Input sequence length
        n_dyn_fea (int): Number of dynamic features
        n_outputs (int): Number of outputs of the network
        kernel_size (int): Kernel size of each convolutional layer
        n_filters (int): Number of filters in each convolutional layer
        dropout_rate (float): Dropout rate in the network
        max_cat_id (list[int]): Each entry in the list represents the maximum value of the ID of a specific categorical variable.

    Returns:
        object: Keras Model object
    """
    # Sequential input for dynamic features
    seq_in = Input(shape=(seq_len, n_dyn_fea))

    # Categorical input
    n_cat_fea = len(max_cat_id)
    cat_fea_in = Input(shape=(n_cat_fea,), dtype="uint8")
    cat_flatten = []
    for i, m in enumerate(max_cat_id):
        cat_fea = Lambda(lambda x, i: x[:, i, None], arguments={"i": i})(cat_fea_in)
        cat_fea_embed = Embedding(m + 1, ceil(log(m + 1)), input_length=1)(cat_fea)
        cat_flatten.append(Flatten()(cat_fea_embed))

    # Dilated convolutional layers
    conv1d_layers = []
    conv1d_layers.append(
        Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=1, padding="causal", activation="relu")(seq_in)
    )
    for i in range(1, n_dilated_layers):
        conv1d_layers.append(
            Conv1D(
                filters=n_filters, kernel_size=kernel_size, dilation_rate=2 ** i, padding="causal", activation="relu"
            )(conv1d_layers[i - 1])
        )

    # Skip connections
    if n_dilated_layers > 1:
        c = concatenate([conv1d_layers[0], conv1d_layers[-1]])
    else:
        c = conv1d_layers[0]

    # Output of convolutional layers
    conv_out = Conv1D(8, 1, activation="relu")(c)
    conv_out = Dropout(dropout_rate)(conv_out)
    conv_out = Flatten()(conv_out)

    # Concatenate with categorical features
    x = concatenate([conv_out] + cat_flatten)
    x = Dense(64, activation="relu")(x)
    df_output = Dense(n_outputs, activation="linear")(x)

    # vlt layer
    vlt_input = Input(shape=(seq_len,))
    vlt_cat_feature = concatenate([Flatten()(vlt_input)] + cat_flatten)
    vlt_layer1 = Dense(16, activation="relu")(vlt_cat_feature)
    vlt_layer1 = Dropout(dropout_rate)(vlt_layer1)
    vlt_layer2 = Dense(1, activation="relu")(vlt_layer1)
    vlt_output = vlt_layer2

    # Reorder point (review period)
    rp_in = Input(shape=(seq_len,))
    layer3_input = concatenate([x, vlt_layer1, rp_in])
    layer3_output = Dense(32, activation="relu")(layer3_input)
    layer3_output = Dropout(dropout_rate)(layer3_output)

    # initial_stock
    initial_stock_in = Input(shape=(seq_len,))
    layer4_input = concatenate([layer3_output, initial_stock_in])
    layer4_hidden = Dense(64, activation="relu")(layer4_input)
    layer4_hidden = Dropout(dropout_rate)(layer4_hidden)
    layer4_output = Dense(1, activation="relu")(layer4_hidden)
    # Define model interface
    model = Model(inputs=[seq_in, cat_fea_in, vlt_input, rp_in, initial_stock_in],
                  outputs=[df_output, layer4_output, vlt_output], loss_weights=[0.05, 0.9, 0.05])

    return model


def custom_loss(y_true, y_pred):
    # Calculate binary cross entropy
    mse = tf.keras.losses.MeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()


    # Calculate loss
    loss = 0.1 * mse(y_true[0], y_pred[0]) + 0.8 * mse(y_true[1], y_pred[1]) + 0.1 * mse(y_true[2], y_pred[2])

    return loss


# def custom_loss(y_true, y_pred):
#     epsilon = 1e-5  # Small constant to avoid division by zero
#     mape = tf.keras.losses.MeanAbsolutePercentageError()
#
#     # Adjusted MAPE for division by zero
#     loss = (
#             0.3 * mape(y_true[0] + epsilon, y_pred[0]) +
#             0.5 * mape(y_true[1] + epsilon, y_pred[1]) +
#             0.2 * mape(y_true[2] + epsilon, y_pred[2])
#     )
#
#     return loss

