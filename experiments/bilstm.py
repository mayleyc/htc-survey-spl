from typing import Dict, Type

from src.datasets.generation.base import BaseDataset
from src.datasets.generation.tl import EnWiki, ItWiki, FrWiki
from src.datasets.generation.nc import RCV1en, RCV2it, RCV2fr
from src.models.bilstms.keras_handler import KerasScript
from src.models.bilstms.models import (RevisedBiLSTM)
from utils.general import load_yaml


def main():
    dataset: Type[BaseDataset] = FrWiki
    print(f"Current language: {dataset.LANG.title()}")
    run_config: Dict = load_yaml('configs/keras_run.yml')
    print(f"Current dataset: {run_config['DATASET_NAME']}")
    keras_script = KerasScript(dataset, run_config)
    keras_script.init(preprocess=run_config['PREPROCESS'],
                      load_embeddings=run_config['LOAD_EMBEDDINGS'])
    keras_script.run_model(RevisedBiLSTM)


if __name__ == "__main__":
    main()
