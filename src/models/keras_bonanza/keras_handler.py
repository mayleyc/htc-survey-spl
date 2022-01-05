import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
# import sklearn.metrics as skmetrics
from gensim.models import keyedvectors
from typing import Dict, Type
import tensorflow as tf

from src.datasets.generation.nc import RCV2it, RCV2fr, RCV1en
from src.datasets.generation.tl import ItWiki, EnWiki, FrWiki
from src.datasets.generation.base import BaseDataset
from src.models.keras_bonanza.keras_utils import preprocess_text, get_data, to_tokenized_tensors
from src.models.keras_bonanza.models import (DPCNN, RevisedBiLSTM, RNNCNN, BaseKerasModel)
from utils.embeddings import load_vectors
from utils.general import load_yaml

# Note: Purely for debugging purposes
MAX_ARTICLE_NUM = 99999999


class KerasScript:
    def __init__(self, dataset: Type[BaseDataset], config: Dict):
        self.dataset: Type[BaseDataset] = dataset
        self.language: str = dataset.LANG
        self.train_x, self.train_y = None, None
        self.val_x, self.val_y = None, None
        self.test_x, self.test_y = None, None
        self.tokenizer = None
        self.embedding_matrix = None
        self.logging: bool = config['LOGGING']
        self.config: Dict = config

        if not config['CUDA']:
            tf.config.experimental.set_visible_devices([], 'GPU')

    def init(self, preprocess: bool = True, load_embeddings: bool = True,
             gensim_embeddings: bool = False):
        # Load data from file
        (self.train_x, self.train_y,
         self.val_x, self.val_y,
         self.test_x, self.test_y) = get_data(self.dataset, name=self.config['DATASET_NAME'])
        # Preprocess text
        if preprocess:
            self._preprocess_text()

        self.tokenizer = Tokenizer(num_words=self.config['MAX_NB_WORDS'],
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                   lower=not preprocess)
        self.tokenizer.fit_on_texts(self.train_x)
        word2int_indices = self.tokenizer.word_index
        print(len(word2int_indices))
        # *** Transform data for learning ***
        self.train_x, self.train_y = to_tokenized_tensors(self.train_x, self.train_y[:MAX_ARTICLE_NUM], self.tokenizer,
                                                          self.config['MAX_SEQUENCE_LENGTH'])
        self.val_x, self.val_y = to_tokenized_tensors(self.val_x, self.val_y[:MAX_ARTICLE_NUM], self.tokenizer,
                                                      self.config['MAX_SEQUENCE_LENGTH'])
        self.test_x, self.test_y = to_tokenized_tensors(self.test_x, self.test_y[:MAX_ARTICLE_NUM], self.tokenizer,
                                                        self.config['MAX_SEQUENCE_LENGTH'])
        if load_embeddings:
            if not gensim_embeddings:
                self._load_embeddings(word2int_indices)
            else:
                self._load_gensim_embeddings(word2int_indices)
        if self.logging:
            print(f'Found {len(word2int_indices)} unique tokens')
            print(f'Shape of train text tensor: {self.train_x.shape}')
            print(f'Shape of train label tensor: {self.train_y.shape}')

    def _load_embeddings(self, word2int_indices: Dict):
        vectors, _ = load_vectors(self.config['EMBEDDING_PATH'][self.dataset.LANG],
                                  oov=False, write_vocab=False, max_vectors=self.config['MAX_FT_VECS'])
        self.embedding_matrix = np.zeros((min(len(word2int_indices) + 1,
                                              self.tokenizer.num_words), vectors.dim))
        for word, i in word2int_indices.items():
            if not (self.tokenizer.num_words and i >= self.tokenizer.num_words):
                self.embedding_matrix[i] = vectors.get_vecs_by_tokens(word).numpy()

    def _load_gensim_embeddings(self, word2int_indices: Dict):
        vecs = keyedvectors.load_word2vec_format(self.dataset.PREFERRED_EMBEDDING_PATH,
                                                 limit=self.config['MAX_VOCABULARY'])
        self.embedding_matrix = np.zeros((len(word2int_indices) + 1, vecs.vector_size))
        for word, i in word2int_indices.items():
            if word in vecs.key_to_index.keys():
                self.embedding_matrix[i] = vecs.vectors[vecs.key_to_index[word]]

    def _preprocess_text(self):
        self.train_x = preprocess_text(self.train_x[:MAX_ARTICLE_NUM], self.dataset)
        self.val_x = preprocess_text(self.val_x[:MAX_ARTICLE_NUM], self.dataset)
        self.test_x = preprocess_text(self.test_x[:MAX_ARTICLE_NUM], self.dataset)

    def run_model(self, keras_model: Type[BaseKerasModel]):
        print(f" - Current language: {self.language.title()} - ")
        keras_net = keras_model(self.train_x.shape[1],
                                self.dataset.NUM_CLASSES,
                                self.embedding_matrix,
                                self.config)
        model = keras_net.get_model()

        model.fit(self.train_x, self.train_y, epochs=self.config['EPOCHS'], batch_size=self.config['BATCH_SIZE'],
                  validation_data=(self.val_x, self.val_y),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001)])
        # *** Evalutate ***
        accr = model.evaluate(self.test_x, self.test_y)
        print(f'Test set\n  Loss: {accr[0]:0.3f}\n  Accuracy: {accr[1]:0.3f}')

        preds = [[round(y) for y in x] for x in model.predict(self.test_x)]

        pre, rec, f1, _ = precision_recall_fscore_support(self.test_y, preds, average="macro")
        acc = accuracy_score(self.test_y, preds)
        metrics = dict(accuracy=acc, precision=pre, recall=rec, f1_score=f1)
        pp = pd.Series(metrics).to_string(dtype=False)
        print("\n*** KERAS test metrics ***")
        print(pp)


# def main():
#     dataset: Type[BaseDataset] = RCV2it
#     print(f"Current language: {dataset.LANG.title()}")
#     run_config: Dict = load_yaml('configs/keras_run.yml')
#     keras_script = KerasScript(dataset, run_config)
#     keras_script.init(preprocess=run_config['PREPROCESS'],
#                       load_embeddings=run_config['LOAD_EMBEDDINGS'])
#     keras_script.run_model(RevisedBiLSTM)
#
#
# if __name__ == "__main__":
#     main()
