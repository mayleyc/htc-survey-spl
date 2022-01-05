from src.datasets.generation.nc.base_rcv import BaseRCV


class RCV2it(BaseRCV):
    NUM_CLASSES: int = 15
    LANG: str = "italian"
    TRAIN_SIZE: float = .6  # Of the overall dataset
    VAL_SIZE: float = .2  # Of the train set
    MAX_CODE_LEN: int = 3  # Code length decides the hierarchy level at which to cut


class RCV2fr(BaseRCV):
    NUM_CLASSES: int = 38
    LANG: str = "french"
    TRAIN_SIZE: float = .6
    VAL_SIZE: float = .2
    MAX_CODE_LEN: int = 3


class RCV1en(BaseRCV):
    NUM_CLASSES: int = 57
    LANG: str = "english"
    TRAIN_SIZE: float = .6
    VAL_SIZE: float = .2
    MAX_CODE_LEN: int = 3


if __name__ == "__main__":
    rcv2_it = RCV2it(dataset_name="rcv2it-2")
    rcv2_it.generate(plot=False)

    rcv2_fr = RCV2fr(dataset_name="rcv2fr-2")
    rcv2_fr.generate(plot=False)

    rcv1_en = RCV1en(dataset_name="rcv1en-2")
    rcv1_en.generate(plot=False)
