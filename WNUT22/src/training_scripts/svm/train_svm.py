from pathlib import Path
from typing import Dict

from src.models.tfidf.tfidf_svm import prepare_train_test_data, classify_flattened
from src.utils.generic_functions import load_yaml


def run_svm():
    config_path = Path("configs") / "svm"
    out_folder: Path = Path("out") / "svm"
    config: Dict = load_yaml(config_path / "config.yml")

    data, labels = prepare_train_test_data(dataset=config["dataset"])
    classify_flattened(data, labels, config, out_folder)


if __name__ == "__main__":
    run_svm()
