import fasttext
import pandas
import pandas as pd

from src.datasets.generation.base import BaseDataset


class SentiPolc(BaseDataset):
    """
    Dataset generator for Sentipolc 2016 dataset. For comparability
    with other SA tasks, only the polarity subtask is kept.
    """

    NUM_CLASSES: int = 2
    MULTILABEL: bool = False

    def __init__(self):
        super().__init__()

    def generate(self, *args, **kwargs) -> None:
        train = pd.read_csv(self.RAW_ROOT / "train.csv", sep=";")
        test = pd.read_csv(self.RAW_ROOT / "test.csv", sep=";", names=list(train.columns))
        print("Hello there")

