from typing import List, Callable, Dict, Tuple, Union

import joblib
import numpy as np
import regex as re
import torch
import torch.utils.data as td
from tokenizers import Tokenizer


class SimplifiedDataset:
    NUM_CLASSES: int = 15

    def __init__(self, dataset_path: str, encoder_path: str):
        self.x = []
        self.y = []
        self.decoder = joblib.load(encoder_path)
        for line in open(dataset_path, 'r', encoding='utf-8').readlines():
            _x, _y = self.__process(line)
            self.x.append(_x)
            self.y.append(_y)

    def __process(self, line: str) -> Tuple[str, np.ndarray]:
        tagged_labels = re.findall(r'__label__\w+', line)
        text = line[len(" ".join(tagged_labels)) + 1:]
        labels = self.decoder.transform([[cat[len('__label__'):] for cat in tagged_labels]])[0]
        return text, labels


class TransformerMapDataset(td.Dataset):
    """
    Dataset for transformer Bert
    """

    def __init__(self, dataset_path: str, encoder_path: str, tokenizer: Union[Callable, Tokenizer]):
        self.x = []
        self.y = []
        self.decoder = joblib.load(encoder_path)
        for line in open(dataset_path, 'r', encoding='utf-8').readlines():
            _x, _y = self.__process(line)
            self.x.append(_x)
            self.y.append(_y)

        self._tokenize(tokenizer)

    def _tokenize(self, tokenizer: Callable) -> None:
        """
        Perform the tokenization step

        :param tokenizer: a callable tokenizer from HF library
        """
        # When not set, it's something like +inf. 2048 is currently the longest transformer available
        max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
        self.x = tokenizer(self.x, truncation=True, max_length=max_len, padding=True)

    def __process(self, line: str) -> Tuple[str, np.ndarray]:
        """
        Process a FastText formatted line into a [text(str), ndarray(binary labels)] data point

        Parameters
        ----------
        line : line string being processed

        Returns
        -------
        text, labels as a string and ndarray of binary values (labels)
        """
        tagged_labels = re.findall(r'__label__\w+', line)
        text = line[len(" ".join(tagged_labels)) + 1:]
        labels = self.decoder.transform([[cat[len('__label__'):] for cat in tagged_labels]])[0]
        return text, labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.x.items()}
        labels = torch.as_tensor(self.y[idx])
        item["labels"] = labels  # F.one_hot(labels, num_classes=20)

        return item

    def __len__(self):
        return len(self.y)


class TransformerIterableDataset(td.IterableDataset):
    def __init__(self, dataset_path: str, encoder_path: str, batch_size: int):
        """
        Iterable dataset for Transformers. Allows to load batches gradually from file.

        Parameters
        ----------
        dataset_path : Path to dataset in FastText (txt) format
        encoder_path : Path to binary encoder (FastText format does NOT allow binarized labels)
        batch_size : size of batches to return. Will determine dataloader size
        """
        self.path: str = dataset_path
        self.__len = None
        self.batch_size = batch_size
        self.decoder = joblib.load(encoder_path)
        self.current_line = 0

    def __process(self, line: str) -> Tuple[str, np.ndarray]:
        """
        Process a FastText formatted line into a [text(str), ndarray(binary labels)] data point

        Parameters
        ----------
        line : line string being processed

        Returns
        -------
        text, labels as a string and ndarray of binary values (labels)
        """
        tagged_labels = re.findall(r'__label__\w+', line)
        text = line[len(" ".join(tagged_labels)) + 1:]
        labels = self.decoder.transform([[cat[len('__label__'):] for cat in tagged_labels]])[0]
        return text, labels

    def __iter__(self):
        """
        Overrides iter method

        Returns
        -------
        yields a batch
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            batch = list()
            for line in f:
                batch.append(self.__process(line))
                if len(batch) == self.batch_size:  # batch full
                    yield batch
                    batch.clear()
                    # gc.collect()
            if batch:  # yield unfull batch
                yield batch
            # -------------------
            # batch = list()
            # line = f.readline()
            # while line:
            #     batch.append(self.__process(line))
            #     if len(batch) == self.batch_size:  # batch full
            #         yield batch
            #         batch.clear()
            #     line = f.readline()
            # if batch:  # yield unfull batch
            #     yield batch
            # -------------------
            # for _, batch in itertools.groupby(f, key=lambda k, line=itertools.count(): next(line) // self.batch_size):
            #     new_batch = copy.deepcopy(batch)
            #     yield [self.__process(x) for x in new_batch]
            #     del new_batch

    def __len__(self) -> int:
        """
        Overrides len method

        Returns
        -------
        Length of dataset (number of batches)
        """
        if self.__len is None:
            with open(self.path, encoding='utf-8') as f:
                tot: int = len(f.readlines())
                # "len" is really the number of batches
                self.__len = tot // self.batch_size + int(tot % self.batch_size > 1)
        return self.__len

    def __getitem__(self, index):
        return super().__getitem__(index)

    @staticmethod
    def collate_function(tokenizer: Callable, batch: List[Tuple[str, np.ndarray]]) -> Dict:
        """
        Handles batches for pytorch. In this case, also utilizes the transformer's tokenizer
        to prepare the batches

        Parameters
        ----------
        tokenizer : Transformer's pretrained tokenizer
        batch : a batch read from file, as a list of [text,labels]

        Returns
        -------
        An item (a torch-ready dict batch)
        """
        x: List[str] = [x for x, _ in batch]  # This warning is fake and wrong :(
        y: List[np.array] = [np.asarray(y) for _, y in batch]
        # y: np.array = np.array(batch.binary_labels.apply(lambda _x: list(json.loads(_x))).to_list())
        # When not set, it's something like +inf. 2048 is currently the longest transformer available
        max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
        encoded_x = tokenizer(x, truncation=True, padding="max_length", max_length=max_len)
        item = {key: torch.tensor(val) for key, val in encoded_x.items()}
        item["labels"] = torch.as_tensor(np.array(y))
        return item


class WordEmbeddingMapDataset(TransformerMapDataset):
    """
    Dataset for transformer Bert
    """

    def __init__(self, dataset_path: str, encoder_path: str, tokenizer: Tokenizer):
        super().__init__(dataset_path, encoder_path, tokenizer)

    def _tokenize(self, tokenizer: Tokenizer) -> None:
        self.x = tokenizer.encode_batch(self.x, add_special_tokens=False, is_pretokenized=False)

    def __getitem__(self, idx):
        item = torch.as_tensor(self.x[idx].ids, dtype=torch.int)
        labels = torch.as_tensor(self.y[idx])

        return item, labels


def iter_coll_fn_embed(tokenizer: Tokenizer, batch: List[Tuple[str, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenization collate function for embedding iterable datasets

    :param tokenizer: tokenizer
    :param batch: batch of input data
    :return: tensors with encoded and tokenized data
    """
    x, y = zip(*batch)
    x = tokenizer.encode_batch(x, add_special_tokens=False, is_pretokenized=False)
    x = torch.as_tensor([xi.ids for xi in x], dtype=torch.int)
    y = torch.as_tensor(y)
    return x, y
