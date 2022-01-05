from typing import Type, List

from src.datasets.generation.nc import RCV2it, RCV1en, RCV2fr
from src.datasets.generation.tl import ItWiki, EnWiki, FrWiki

start = 1
N = 4

if __name__ == "__main__":
    datasets: List[Type] = [ItWiki, EnWiki, FrWiki, RCV2it, RCV1en, RCV2fr]

    for i in range(start, start + N):
        for dataset in datasets:
            c = dataset(dataset_name=f"{dataset.__name__.lower()}-{i}")
            c.generate(plot=False)
