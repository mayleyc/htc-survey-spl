from src.datasets.generation.tl.base_wiki import BaseWiki


class EnWiki(BaseWiki):
    NUM_CLASSES: int = 100
    LANG: str = "english"
    TRAIN_SIZE: float = .6
    VAL_SIZE: float = .2


if __name__ == "__main__":
    c = EnWiki(dataset_name="enwiki")
    c.generate(plot=False)
