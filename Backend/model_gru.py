import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    Layer,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2  

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="att_weight"
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="att_bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.matmul(inputs, self.W) + self.b
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context

    def get_config(self):
        return super().get_config()


def build_gru_attention_model(
    time_steps: int,
    feature_dim: int,
    gru_units: int = 128,
    dropout: float = 0.4,
):


    inputs = Input(shape=(time_steps, feature_dim))

    x = BatchNormalization()(inputs)
    x = GRU(
        gru_units,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(0.001) 
    )(x)

    x = Dropout(dropout)(x)
    x = AttentionLayer()(x)

    x = Dense(
        64, 
        activation="relu",
        kernel_regularizer=l2(0.001) 
    )(x)
    
    x = Dropout(0.2)(x) 

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model