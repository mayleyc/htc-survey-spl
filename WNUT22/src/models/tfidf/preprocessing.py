import string
from typing import List
from typing import Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from stop_words import get_stop_words

from src.utils.text_utilities.preprocess import text_cleanup


def remove_stopwords(tokens: List[str], min_len: int = 2):
    STOPLIST = set(stopwords.words("english") + get_stop_words("english"))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "…", """, """,
                                                         "–", '-', "...", "<", ">"]
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    tokens = [tok for tok in tokens if len(tok) > min_len]

    return tokens


def process_flattened_dataset(df: pd.DataFrame, remove_garbage: bool = False, stop_words_removal: bool = False):
    # scrap labels, sublabels
    data, _, _, flattened_labels = deeptriage_processing(df, remove_garbage, stop_words_removal)
    return data, flattened_labels


def deeptriage_processing(df: pd.DataFrame, remove_garbage: bool = False, stop_words_removal: bool = False) -> Tuple[
    List[List[str]], List[str], List[str], List[str]]:
    """
    Apply preprocessing as in Deeptriage.

    :param df: dataframe of filtered data
    :param remove_garbage: flag to apply special preproc
    :param stop_words_removal: remove stopwords
    :return: tuple of (processed data (list of tokenized docs), labels, sub-labels, flattened-labels).
    """
    print("Using deeptriage standard preprocessing")
    msg = text_cleanup(df["message"], remove_garbage)
    # Tokenize
    msg_tokens = msg.map(nltk.word_tokenize)
    # Strip punctuation marks
    msg_tokens = msg_tokens.map(lambda tokens: [t.strip(string.punctuation) for t in tokens])
    if stop_words_removal:
        msg_tokens = msg_tokens.map(lambda tokens: remove_stopwords(tokens))
    # Sanity check
    all_tickets_tokens = msg_tokens.tolist()
    all_labels = df["label"].tolist()
    all_sub_labels = df["sub_label"].tolist()
    flattened_labels = df["flattened_label"].tolist()

    assert len(all_tickets_tokens) == len(all_labels) == len(all_sub_labels)
    # Return all types of labels
    return all_tickets_tokens, all_labels, all_sub_labels, flattened_labels
