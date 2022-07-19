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


# # Step 3 - Lemmatize
# def lemmatize(tokens: List[List[str]], allowed_postags=None, logging=False):
#     """
#     Lemmatize (transform into canonical/dictionary form) the current list of tokens
#     :param logging: whether to print pipeline stage
#     :param tokens: List[List[str]] list of list of tokens
#     :param allowed_postags: only the words with the selected Part-of-Speech tags will be kept
#     :return: lemmatized documents
#     """
#     if logging:
#         print("- Current step: Lemmatization.")
#     if allowed_postags is None:
#         allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
#     # Initialize spacy model, keeping only tagger component (for efficiency)
#     # Can also use small models if necessary (end with '_sm' over '_lg')
#     # Packages need to be downloaded: python -m spacy download `model_name`
#     nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
#     # https://spacy.io/api/annotation
#     docs_out = []
#     for sent in tqdm(tokens, desc="Lemmatizing docs"):
#         doc = nlp(" ".join(sent))
#         lemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
#         docs_out.append(lemmatized)
#     return docs_out
#
#
# # Step 3.5 (optional) - Stemming
# def stemming(tokens: List[List[str]], lang: str, logging=False):
#     """
#     Optional step - stemming. Reduce words to root form
#     :param logging: whether to print pipeline stage
#     :param tokens: List[List[str]] list of list of tokens
#     :param lang: language
#     :return: tokens reduced to root form
#     """
#     if logging:
#         print("- Current step: Stemming.")
#     stemmer = SnowballStemmer(language=lang)
#     docs_out = []
#     for toks in tqdm(tokens, desc="Stemming docs"):
#         docs_out.append([stemmer.stem(tok) for tok in toks])
#     return docs_out


def process_flattened_dataset(df: pd.DataFrame, remove_garbage: bool = False,  stop_words_removal: bool = False):
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
