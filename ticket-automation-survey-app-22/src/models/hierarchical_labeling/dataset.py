import pandas as pd
import torch
import torch.utils.data as td
from torchtext.vocab import build_vocab_from_iterator

from src.utils.text_utilities.preprocess import text_cleanup


class TransformerDatasetFlat(td.Dataset):
    """
    Dataset for Transformers with flattened labels
    """

    def __init__(self, data: pd.Series, labels: pd.Series, remove_garbage: bool = False):
        super().__init__()

        self.x = data
        self.y = labels.tolist()
        self.n_y = len(set(self.y))
        self._size = len(self.x)
        self.prepare_dataset(remove_garbage)

    def prepare_dataset(self, remove_garbage: bool = False):
        self.x = text_cleanup(self.x, remove_garbage=remove_garbage)
        self.x = self.x.tolist()

        voc1 = build_vocab_from_iterator([self.y])

        y = voc1(self.y)

        self.y = torch.LongTensor(y)

    def __getitem__(self, idx):
        item = self.x[idx]
        label = self.y[idx]
        return item, label

    def __len__(self) -> int:
        return self._size


class TransformerDataset2Levels(td.Dataset):
    """
    Dataset for Transformers that require two level of labels
    """

    def __init__(self, data: pd.Series, labels: pd.DataFrame, remove_garbage: bool = False):
        super().__init__()

        self.x = data
        self.y1 = labels["label"].tolist()
        self.y = labels["flattened_label"].tolist()
        self.n_y1 = len(set(self.y1))
        self.n_y = len(set(self.y))
        self._size = len(self.x)
        self.prepare_dataset(remove_garbage)

    def prepare_dataset(self, remove_garbage: bool = False):
        self.x = text_cleanup(self.x, remove_garbage=remove_garbage)
        self.x = self.x.tolist()

        voc1 = build_vocab_from_iterator([self.y1])
        voc2 = build_vocab_from_iterator([self.y])

        y1 = voc1(self.y1)
        y2 = voc2(self.y)

        self.y1 = torch.LongTensor(y1)
        self.y = torch.LongTensor(y2)

    def __getitem__(self, idx):
        item = self.x[idx]
        label_1 = self.y1[idx]
        label_2 = self.y[idx]
        return item, label_1, label_2

    def __len__(self) -> int:
        return self._size
