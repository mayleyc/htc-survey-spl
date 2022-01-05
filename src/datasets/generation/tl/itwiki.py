from src.datasets.generation.tl.base_wiki import BaseWiki


class ItWiki(BaseWiki):
    NUM_CLASSES: int = 100
    LANG: str = "italian"
    TRAIN_SIZE: float = .6
    VAL_SIZE: float = .2


if __name__ == "__main__":
    c = ItWiki(dataset_name="itwiki")
    c.generate(plot=False)
