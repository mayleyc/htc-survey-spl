import string
from typing import List

import gensim
import spacy
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from tqdm import tqdm


# Step 0 - receive list of docs
# Step 1 - tokenize
def tokenize(docs: List[str], logging=False):
    """
    Transform a list of docs (strings) into a list of tokens (words)

    :param logging: whether to print pipeline stage
    :param docs: List[str] of docs
    :return: List[List[str]] list of list of tokens
    """
    if logging:
        print("Current step: Tokenization.")
    tokens = []
    for doc in docs:
        # deacc=True removes punctuations
        tokens.append(gensim.utils.simple_preprocess(doc, deacc=True))
    return tokens


# Step 2 (optional but suggested) - Find n-grams
def build_ngrams(tokens: List[List[str]], logging=False):
    """
    Build ngram models for the current list of tokens. This just creates the model and returns them,
    they then have to applied through make_bigrams/make_trigrams

    :param logging: whether to print pipeline stage
    :param tokens: List[List[str]] list of list of tokens
    :return: bigram and trigram model
    """
    if logging:
        print("Current step: N-gram discovery.")
    # Higher threshold fewer phrases.
    bigram = Phrases(tokens, min_count=10, threshold=200)
    trigram = Phrases(bigram[tokens], threshold=200)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    return bigram_mod, trigram_mod


def make_bigrams(bigram_mod, tokens: List[List[str]]):
    """
    Uses bigram to attempt to create bigrams. Returns a list of tokens where tokens
    deemed related are fused into 2grams

    :param bigram_mod: bigram model previously created
    :param tokens: List[List[str]] list of list of tokens
    :return: tokens, where components of a 2gram have been fused into a word_word 2gram
    """
    return [bigram_mod[doc] for doc in tokens]


def make_trigrams(bigram_mod, trigram_mod, tokens: List[List[str]]):
    """
    Uses bigram and trigram model to attempt to create trigrams. Returns a list of tokens where tokens
    deemed related are fused into 3grams

    :param bigram_mod: bigram model previously created
    :param trigram_mod: trigram model previously created
    :param tokens: List[List[str]] list of list of tokens
    :return: tokens, where components of a 3gram have been fused into a word_word_word 3gram
    """
    return [trigram_mod[bigram_mod[doc]] for doc in tokens]


# Step 3 - Lemmatize
def lemmatize(tokens: List[List[str]], lang: str, allowed_postags=None, logging=False):
    """
    Lemmatize (transform into canonical/dictionary form) the current list of tokens

    :param logging: whether to print pipeline stage
    :param tokens: List[List[str]] list of list of tokens
    :param allowed_postags: only the words with the selected Part-of-Speech tags will be kept
    :param lang: language
    :return: lemmatized documents
    """
    if logging:
        print("Current step: Lemmatization.")
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    # Initialize spacy model, keeping only tagger component (for efficiency)
    # Can also use small models if necessary (end with '_sm' over '_lg')
    # Packages need to be downloaded: python -m spacy download `model_name`
    if lang == 'italian':
        nlp = spacy.load('it_core_news_lg', disable=['parser', 'ner'])
    elif lang == 'english':
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    elif lang == 'french':
        nlp = spacy.load('fr_core_news_lg', disable=['parser', 'ner'])
    else:
        raise ValueError('Invalid language code')
    # https://spacy.io/api/annotation
    docs_out = []
    for sent in tqdm(tokens, desc="Lemmatizing docs"):
        doc = nlp(" ".join(sent))
        lemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        docs_out.append(lemmatized)
    return docs_out


# Step 3.5 (optional) - Stemming
def stemming(tokens: List[List[str]], lang: str, logging=False):
    """
    Optional step - stemming. Reduce words to root form

    :param logging: whether to print pipeline stage
    :param tokens: List[List[str]] list of list of tokens
    :param lang: language
    :return: tokens reduced to root form
    """
    if logging:
        print("Current step: Stemming.")
    stemmer = SnowballStemmer(language=lang)
    docs_out = []
    for toks in tqdm(tokens, desc="Stemming docs"):
        docs_out.append([stemmer.stem(tok) for tok in toks])
    return docs_out


# Step 4 - Remove stop words
def remove_stopwords(tokens: List[List[str]], lang: str,
                     min_len: int = 2, logging=False):
    """
    Remove stopwords from list of tokens

    Parameters
    ----------
    tokens : List of tokenized sentences
    lang : language
    min_len : minimum length of tokens to keep
    logging : whether to print results

    Returns
    -------
    Results pruned of stopwords
    """
    if logging:
        print("Current step: Stopwords removal.")
    STOPLIST = set(stopwords.words(lang) + get_stop_words(lang))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "…", """, """,
                                                         "–", '-', "...", "<", ">"]
    clean_tokens = []
    for token_list in tokens:
        token_list = [tok for tok in token_list
                      if tok not in STOPLIST]
        token_list = [tok for tok in token_list
                      if tok not in SYMBOLS]
        token_list = [tok for tok in token_list
                      if len(tok) > min_len]
        clean_tokens.append(token_list)

    return clean_tokens


def pipeline(docs: List[str], lang: str, disable=None, allowed_postags=None, logging=False):
    if disable is None:
        disable = ['trigrams']
    # Tokenize
    tokens = tokenize(docs, logging=logging)
    # Build ngram models
    if 'bigrams' not in disable or 'trigrams' not in disable:
        bi_mod, tri_mod = build_ngrams(tokens, logging=logging)
    # Remove stopwords
    tokens = remove_stopwords(tokens, logging=logging, lang=lang)
    # (Optional) find 2/3-grams
    if 'bigrams' not in disable:
        tokens = make_bigrams(bi_mod, tokens)
    if 'trigrams' not in disable:
        tokens = make_trigrams(bi_mod, tri_mod, tokens)
    # Lemmatize to base form
    if 'lemma' not in disable:
        tokens = lemmatize(tokens, lang, allowed_postags, logging=logging)
    if 'stem' not in disable:
        tokens = stemming(tokens, lang)
    # Second pass of stopword removal
    tokens = remove_stopwords(tokens, logging=logging, lang=lang)
    return tokens
