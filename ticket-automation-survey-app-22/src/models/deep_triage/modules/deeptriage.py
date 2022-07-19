from typing import Dict

from tensorflow import keras
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Input, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.deep_triage.modules.soft_attention_concat import SoftAttentionConcat


class DeepTriage:
    # Credit: https://bugtriage.mybluemix.net/
    def __init__(self, config: Dict, num_classes: int, vocab_size: int, embeddings: np.ndarray):
        self.config = config
        max_sentence_len = config["max_sentence_len"]
        embed_size = config["embed_size"]
        dense_dim = config["dense_dim"]
        lstm_dim = config["lstm_dim"]

        # Construct the architecture for deep bidirectional RNN model using Keras library
        input_layer = Input(shape=(max_sentence_len,), dtype="int32")

        sequence_embed = Embedding(vocab_size,
                                   embed_size,
                                   input_length=max_sentence_len,
                                   embeddings_initializer=Constant(embeddings)
                                   )(input_layer)

        forwards_1 = LSTM(lstm_dim, return_sequences=True, recurrent_dropout=0.2)(sequence_embed)
        attention_1 = SoftAttentionConcat()(forwards_1)
        after_dp_forward_5 = BatchNormalization()(attention_1)

        backwards_1 = LSTM(lstm_dim, return_sequences=True, recurrent_dropout=0.2, go_backwards=True)(sequence_embed)
        attention_2 = SoftAttentionConcat()(backwards_1)
        after_dp_backward_5 = BatchNormalization()(attention_2)

        merged = Concatenate(axis=-1)([after_dp_forward_5, after_dp_backward_5])
        after_merge = Dense(dense_dim, activation="relu")(merged)
        after_dp = Dropout(config["dropout"])(after_merge)
        output = Dense(num_classes, activation="softmax")(after_dp)
        self.model = Model(inputs=input_layer, outputs=output)

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.config["lr"]),
                           metrics=["accuracy"])

    def summary(self):
        self.model.summary()
