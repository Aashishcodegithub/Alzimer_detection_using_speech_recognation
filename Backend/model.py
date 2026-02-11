from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    LSTM,
    Dense
)
from tensorflow.keras.regularizers import l2

def build_cnn_lstm_model(
    time_steps: int,
    feature_dim: int,
    lstm_units: int = 32, 
    dropout: float = 0.5,
):
    inputs = Input(shape=(time_steps, feature_dim))

    x = BatchNormalization(name="input_batchnorm")(inputs)
    x = Conv1D(
        filters=32, 
        kernel_size=5, 
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(
        filters=64, 
        kernel_size=3, 
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = LSTM(
        lstm_units,
        return_sequences=False,
        dropout=0.6,          
        recurrent_dropout=0.3,
        kernel_regularizer=l2(0.01) 
    )(x)

    outputs = Dense(
        1, 
        activation="sigmoid",
        kernel_regularizer=l2(0.001)
    )(x)

    return Model(inputs, outputs)