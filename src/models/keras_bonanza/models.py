import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, SpatialDropout1D, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, LSTM, Lambda, Conv1D, GRU, Dense, Bidirectional
from tensorflow.keras.layers import Activation, Add, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Dict

# The maximum number of words to be used. (most frequent)
# MAX_NB_WORDS = 87897  # AKA max features


class BaseKerasModel:
    def __init__(self, shape, num_classes, embedding_matrix, config: Dict):
        self.model = None
        self.input_length = shape
        self.num_classes = num_classes
        self.embedding_matrix = embedding_matrix

        self.config = config
        self.loss = 'binary_crossentropy'
        self.opt = Adam(learning_rate=self.config['LEARNING_RATE'])

    def get_model(self):
        print(self.model.summary())
        return self.model


class TextBaseLSTM(BaseKerasModel):
    def __init__(self, shape, num_classes, embedding_matrix, config):
        super().__init__(shape, num_classes, embedding_matrix, config)
        self.model = Sequential()
        self.model.add(Embedding(self.config['MAX_NB_WORDS'] if embedding_matrix is None
                                 else self.embedding_matrix.shape[0],
                                 self.config['EMBEDDING_DIM'] if embedding_matrix is None
                                 else embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix] if embedding_matrix is not None else None,
                                 trainable=self.config['WEIGHTS_TRAINABLE'],
                                 input_length=self.input_length))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])

    def get_model(self):
        print(self.model.summary())
        return self.model


class BiLSTM(BaseKerasModel):
    def __init__(self, shape, num_classes, embedding_matrix, config):
        super().__init__(shape, num_classes, embedding_matrix, config)
        self.model = Sequential()
        self.model.add(Embedding(self.config['MAX_NB_WORDS'] if embedding_matrix is None
                                 else self.embedding_matrix.shape[0],
                                 self.config['EMBEDDING_DIM'] if embedding_matrix is None
                                 else embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix] if embedding_matrix is not None else None,
                                 trainable=self.config['WEIGHTS_TRAINABLE'],
                                 input_length=self.input_length))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])


class RevisedBiLSTM(BaseKerasModel):
    def __init__(self, shape, num_classes, embedding_matrix, config):
        super().__init__(shape, num_classes, embedding_matrix, config)
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.config['MAX_NB_WORDS'] if embedding_matrix is None
                                 else self.embedding_matrix.shape[0],
                                 output_dim=self.config['EMBEDDING_DIM'] if embedding_matrix is None
                                 else embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix] if embedding_matrix is not None else None,
                                 trainable=self.config['WEIGHTS_TRAINABLE'],
                                 input_length=self.input_length))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(300, return_sequences=True)))
        global_max_pooling = Lambda(lambda x: K.max(x, axis=1))
        self.model.add(global_max_pooling)  # GlobalMaxPooling1D didn't support masking
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])


class RNNCNN(BaseKerasModel):
    def __init__(self, shape, num_classes, embedding_matrix, config):
        super().__init__(shape, num_classes, embedding_matrix, config)

        input_text = Input(shape=(shape,))
        embedding_layer = Embedding(input_dim=self.config['MAX_NB_WORDS'] if embedding_matrix is None
                                    else self.embedding_matrix.shape[0],
                                    output_dim=self.config['EMBEDDING_DIM'] if embedding_matrix is None
                                    else embedding_matrix.shape[1],
                                    weights=[self.embedding_matrix] if embedding_matrix is not None else None,
                                    trainable=self.config['WEIGHTS_TRAINABLE'],
                                    input_length=self.input_length)(input_text)
        # embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=shape)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)
        gru_layer = Bidirectional(GRU(300, return_sequences=True))(text_embed)
        conv_layer = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(gru_layer)
        avg_pool = GlobalAveragePooling1D()(conv_layer)
        max_pool = GlobalMaxPooling1D()(conv_layer)
        sentence_embed = concatenate([avg_pool, max_pool])
        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.num_classes, activation='sigmoid')(dense_layer)
        self.model = Model(input_text, output)
        self.model.compile(loss=self.loss, metrics=['accuracy'], optimizer=self.opt)


class DPCNN(BaseKerasModel):
    def __init__(self, shape, num_classes, embedding_matrix, config):
        super().__init__(shape, num_classes, embedding_matrix, config)
        input_text = Input(shape=(shape,))
        embedding_layer = Embedding(input_dim=self.config['MAX_NB_WORDS'] if embedding_matrix is None
                                    else self.embedding_matrix.shape[0],
                                    output_dim=self.config['EMBEDDING_DIM'] if embedding_matrix is None
                                    else embedding_matrix.shape[1],
                                    weights=[self.embedding_matrix] if embedding_matrix is not None else None,
                                    trainable=self.config['WEIGHTS_TRAINABLE'],
                                    input_length=self.input_length)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        repeat = 3
        size = shape
        region_x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(text_embed)
        x = Activation(activation='relu')(region_x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Activation(activation='relu')(x)
        x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
        x = Add()([x, region_x])

        for _ in range(repeat):
            px = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
            size = int((size + 1) / 2)
            x = Activation(activation='relu')(px)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Activation(activation='relu')(x)
            x = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)(x)
            x = Add()([x, px])

        x = MaxPooling1D(pool_size=size)(x)
        sentence_embed = Flatten()(x)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.num_classes, activation='sigmoid')(dense_layer)
        self.model = Model(input_text, output)
        self.model.compile(loss=self.loss, metrics=['accuracy'], optimizer=self.opt)
