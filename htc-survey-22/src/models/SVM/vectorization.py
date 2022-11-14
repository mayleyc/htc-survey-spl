import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace

from src.models.Capsules.dataset import load_vectors
from src.models.SVM.preprocessing import process_list_dataset
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

def tfidf_processing(train_x, test_x, config):
    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    x_train, x_test = process_list_dataset(train_x, test_x,
                                           remove_garbage=config["remove_garbage"],
                                           stop_words_removal=config["stop_words_removal"])
    print("Preprocessing complete.\n")
    # Create feature vectors
    vectorizer = TfidfVectorizer(max_features=config["MAX_FEATURES"], ngram_range=(1, 2))
    # Train the feature vectors
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    return train_vectors, test_vectors


def glove_processing(train_x, test_x, config):
    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    x_train, x_test = process_list_dataset(train_x, test_x,
                                           remove_garbage=config["remove_garbage"],
                                           stop_words_removal=config["stop_words_removal"], join=True)
    print("Preprocessing complete.\n")
    # -----------------------------------------------------------
    print("Loading Glove embeddings:")
    vectors, vocab = load_vectors(config["pretrained_embedding"], max_vectors=500000)
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = BertNormalizer(strip_accents=True, lowercase=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=config["NUM_TOKENS"])

    x_train = tokenizer.encode_batch(x_train, add_special_tokens=False, is_pretokenized=False)
    x_test = tokenizer.encode_batch(x_test, add_special_tokens=False, is_pretokenized=False)
    x_train = [vectors.vectors[x.ids].numpy() for x in x_train]
    x_test = [vectors.vectors[x.ids].numpy() for x in x_test]

    return x_train, x_test


def fasttext_processing(train_x, test_x, config):
    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    x_train, x_test = process_list_dataset(train_x, test_x,
                                           remove_garbage=config["remove_garbage"],
                                           stop_words_removal=config["stop_words_removal"])
    print("Preprocessing complete.\n")
    kv = KeyedVectors.load_word2vec_format(config["pretrained_embedding"], limit=config["MAX_FEATURES"])
    # Add element for unknown terms
    kv.vectors = np.concatenate([kv.vectors, np.zeros((1, kv.vectors[0].shape[0]))])
    kv.index_to_key.append("<unk>")
    kv.key_to_index["<unk>"] = len(kv.vectors) - 1
    tokenizer = Tokenizer(WordLevel(kv.key_to_index, unk_token="<unk>"))
    tokenizer.normalizer = BertNormalizer(strip_accents=True, lowercase=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=config["NUM_TOKENS"])

    x_train = tokenizer.encode_batch(x_train, add_special_tokens=False, is_pretokenized=False)
    x_test = tokenizer.encode_batch(x_test, add_special_tokens=False, is_pretokenized=False)
    x_train = [kv.vectors[x.ids] for x in x_train]
    x_test = [kv.vectors[x.ids] for x in x_test]
    return x_train, x_test
