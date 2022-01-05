import abc
import os
from pathlib import Path


class BaseDataset(abc.ABC):
    """
    Base class for dataset generation. This class is responsible for producing the dataset from its raw files.
    It should also split into train and test set.
    """
    NUM_CLASSES: int
    MULTILABEL: bool
    LANG: str
    PREFERRED_EMBEDDING_PATH: str

    def __init__(self, dataset_name: str):
        """
        Abstract constructor for dataset base class

        :param dataset_name: the name of the dataset folder where to generate splits
        """
        # path to RAW dataset files
        self.RAW_ROOT: Path = self.raw_root()

        if not os.path.isdir(self.RAW_ROOT):
            raise NotADirectoryError("RAW dataset directory is not valid")

        # path where to generate the final dataset files
        self.GENERATED_ROOT: Path = BaseDataset.generated_root(dataset_name)
        self.ENCODER_FILEPATH: Path = self.GENERATED_ROOT / "encoder.bin.xz"

        if not os.path.isdir(self.GENERATED_ROOT):
            os.makedirs(self.GENERATED_ROOT, exist_ok=True)

    @staticmethod
    def generated_root(name: str) -> Path:
        return Path("data") / "generated" / name

    @classmethod
    def raw_root(cls) -> Path:
        return Path("data") / "raw" / cls.__name__.lower()

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> None:
        """
        Start generation. This method must also train a LabelEncoder to be used for encoding labels to numbers
        or one-hot vectors
        """
        pass

    def _save_to_txt(self, x_train, x_test, x_val,
                     y_train, y_test, y_val):
        """
        Save to a txt file compliant with FastText rulings

        Parameters
        ----------
        x_train, x_test, x_val, y_train, y_test, y_val : train / test / validation splits
        """
        # Save splits in txt format (FastText compliant)
        with open(self.GENERATED_ROOT / "train.txt", "w+", encoding='utf-8') as train_f:
            lines = list()
            for i in range(len(x_train)):
                lines.append(f"{' '.join([f'__label__{topic}' for topic in y_train[i]])} {x_train[i]}\n")
            train_f.writelines(lines)

        with open(self.GENERATED_ROOT / "test.txt", "w+", encoding='utf-8') as test_f:
            lines = list()
            for i in range(len(x_test)):
                lines.append(f"{' '.join([f'__label__{topic}' for topic in y_test[i]])} {x_test[i]}\n")
            test_f.writelines(lines)

        with open(self.GENERATED_ROOT / "val.txt", "w+", encoding='utf-8') as val_f:
            lines = list()
            for i in range(len(x_val)):
                lines.append(f"{' '.join([f'__label__{topic}' for topic in y_val[i]])} {x_val[i]}\n")
            val_f.writelines(lines)
