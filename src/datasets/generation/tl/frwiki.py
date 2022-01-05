from src.datasets.generation.tl.base_wiki import BaseWiki


class FrWiki(BaseWiki):
    NUM_CLASSES: int = 100
    LANG: str = "french"
    TRAIN_SIZE: float = .6
    VAL_SIZE: float = .2


if __name__ == "__main__":
    c = FrWiki(dataset_name="frwiki")
    c.generate(plot=False)
