import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import src.models.classic_classifiers.text_preprocessing as txt_preproc
from src.datasets.torch_dataset import SimplifiedDataset


def get_data(dataset, name: str):
    train = SimplifiedDataset(os.path.join(dataset.generated_root(name), "train.txt"),
                              os.path.join(dataset.generated_root(name), "encoder.bin.xz"))
    test = SimplifiedDataset(os.path.join(dataset.generated_root(name), "test.txt"),
                             os.path.join(dataset.generated_root(name), "encoder.bin.xz"))
    val = SimplifiedDataset(os.path.join(dataset.generated_root(name), "val.txt"),
                            os.path.join(dataset.generated_root(name), "encoder.bin.xz"))
    train_x = train.x
    train_y = train.y

    val_x = val.x
    val_y = val.y

    test_x = test.x
    test_y = test.y

    return train_x, train_y, val_x, val_y, test_x, test_y


def preprocess_text(x, dataset):
    prep_x = txt_preproc.pipeline(x, dataset.LANG,
                                  disable=['bigrams', 'stem', 'trigrams', 'lemma'], logging=True)
    prep_x = [" ".join(sentence) for sentence in prep_x]
    return prep_x


def to_tokenized_tensors(x, y, tokenizer, max_sequence_length: int):
    # *** Transform data for learning ***
    _x = tokenizer.texts_to_sequences(x)
    _x = pad_sequences(_x, maxlen=max_sequence_length)
    _y = np.stack(y)
    return _x, _y
