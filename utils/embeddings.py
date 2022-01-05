import json
import os
from pathlib import Path
from typing import Union, Tuple, Dict

import torch as th
from torchtext.vocab import Vectors


def load_vectors(path: Union[Path, str], oov: bool = True, max_vectors: int = None, write_vocab: bool = True) \
        -> Tuple[Vectors, Dict[str, int]]:
    """
    Load embeddings and create vocabulary file

    :param path: path to embeddings file
    :param oov: whether to add an out-of-vocabulary token and corresponding zero tensor
    :param max_vectors: max num of vectors to load
    :param write_vocab: whether to write a vocabulary file with token to index mapping
    :return: embeddings in torchtext format and the dictionary of tokens to indexes
    """
    name = os.path.basename(path)
    cache = os.path.dirname(path)
    vec = Vectors(name=name, cache=cache, max_vectors=max_vectors)
    if oov:
        # Add element for unknown terms
        vec.vectors = th.cat([vec.vectors, th.zeros((1, vec.vectors[0].shape[0]))], dim=0)
        vec.itos.append("<unk>")
        vec.stoi["<unk>"] = len(vec) - 1

    if write_vocab:
        vocab_file = os.path.join(cache, f"{name}.vocab.json")
        with open(vocab_file, "w+") as f:
            json.dump(vec.stoi, f)

    return vec, vec.stoi
