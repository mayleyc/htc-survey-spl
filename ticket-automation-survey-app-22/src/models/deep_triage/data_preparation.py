import string
from typing import List, Dict, Tuple

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.dataset_tools.prepare_financial_dataset import read_dataset as read_financial
from src.dataset_tools.prepare_linux_dataset import read_dataset as read_bugs
from src.utils.text_utilities.preprocess import text_cleanup

np.random.seed(1337)

def prepare_for_model(data: List[List[str]], labels: List[str], labels_set: List[str], w2v_model: Word2Vec,
                      config: Dict, embeddings: bool = False):
    max_sentence_len = config["max_sentence_len"]
    embed_size_model = config["embed_size"]

    x_shape = [len(data), max_sentence_len]
    if embeddings is True:
        x_shape.append(embed_size_model)

    x = np.empty(shape=x_shape, dtype="float32")
    y = np.empty(shape=[len(labels), 1], dtype="int32")
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
    for ticket_index, ticket_text in enumerate(data):
        word_sequence_index = 0
        for word in ticket_text:
            if word in w2v_model.wv.key_to_index:
                if embeddings is True:
                    x[ticket_index, word_sequence_index, :] = w2v_model.wv[word]
                else:
                    x[ticket_index, word_sequence_index] = w2v_model.wv.key_to_index[word]
                word_sequence_index += 1
                if word_sequence_index == max_sentence_len - 1:
                    break
        if embeddings is True:
            for k in range(word_sequence_index, max_sentence_len):
                x[ticket_index, k, :] = np.zeros((1, embed_size_model))
        y[ticket_index, 0] = labels_set.index(labels[ticket_index])

    y = to_categorical(y, len(labels_set))
    return x, y


def process_dataset(df: pd.DataFrame, remove_garbage: bool) -> Tuple[List[List[str]], List[str], List[str], List[str]]:
    # df:  message, label, sub_labels

    msg = text_cleanup(df["message"], remove_garbage=remove_garbage)
    # 6. Tokenize
    msg_tokens = msg.map(nltk.word_tokenize)
    # 7. Strip punctuation marks
    msg_tokens = msg_tokens.map(lambda tokens: [t.strip(string.punctuation) for t in tokens])
    # 8. Join the lists (NOT NEEDED)

    all_tickets_tokens = msg_tokens.tolist()
    level_1_labels = df["label"].tolist()
    level_2_labels = df["sub_label"].tolist()
    flattened_labels = df["flattened_label"].tolist()

    assert len(all_tickets_tokens) == len(level_1_labels) == len(level_2_labels) == len(flattened_labels)
    return all_tickets_tokens, level_1_labels, level_2_labels, flattened_labels


def create_embedding_model(tickets: List[List[str]], config: Dict) -> Word2Vec:
    print("Training Embeddings...")
    word_2_vec_model = Word2Vec(tickets, min_count=config["min_word_frequency"], vector_size=config["embed_size"],
                                epochs=config["word2vec_epochs"])
    return word_2_vec_model


def prepare_train_test_data(dataset: str, config) -> Tuple[Word2Vec, List[List[str]], List[str], List[str], List[str]]:
    if dataset == "financial":
        df = read_financial()
    else:
        df = read_bugs()

    print("Starting preprocessing ...")
    data, labels1, labels2, flat_labels = process_dataset(df, remove_garbage=config["REMOVE_GARBAGE_TEXT"])
    print("Preprocessing complete.\n")
    print("Creating embedding model ...")
    w2v = create_embedding_model(data, config)
    print("Embedding model created.")
    return w2v, data, labels1, labels2, flat_labels
