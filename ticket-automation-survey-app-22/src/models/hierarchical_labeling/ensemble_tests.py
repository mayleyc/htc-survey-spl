import os
import re
from pathlib import Path
from typing import Optional

from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.models.hierarchical_labeling.modules.ensembles import EnsembleBERT, SupportedBERT, E2EbiBERT
from src.models.hierarchical_labeling.training_bert4c import run

# INSTRUCTIONS: set the L2 model that you want to use, and the epoch number of the dump to be used (3 typically)
L2_MODEL = Path("src/models/hierarchical_labeling/dumps/bert_financial_l2")
EPOCH = 2


def get_actual_dump_name(dump_root, regex):
    model_dump = None
    for root, dirs, files in os.walk(dump_root):
        for folder in dirs:
            if regex.match(folder):
                model_dump = Path(root) / folder
                break
        if model_dump is not None:
            break
    return model_dump


def ensemble_args_split(dump_folder: Path, e: int, model2: Optional[Path], fold_n: int):
    m1_ = get_actual_dump_name(dump_folder, re.compile(f"fold_{fold_n}.*"))
    model_1 = str(m1_ / f"epoch_{e}" / "model.pt")
    if model2 is not None:
        model_2 = str(get_actual_dump_name(model2, re.compile(f"fold_{fold_n}.*")) / f"epoch_{e}" / "model.pt")
        return dict(MODEL_L1=model_1, MODEL_L2=model_2)
    return dict(MODEL_L1=model_1)


def train_l1_model():
    full_path = Path("src/models/hierarchical_labeling/configs/financial_ensembles")
    file_name = "bert_financial_l1.yml"
    run(file_name, BERTForClassification, full_path=full_path, dataset="financial")
    # config = load_yaml(p / file_name)
    # return Path(config["MODEL_FOLDER"])


def train_l2_model():
    full_path = Path("src/models/hierarchical_labeling/configs/financial_ensembles")
    file_name = "bert_financial_l2.yml"
    run(file_name, BERTForClassification, full_path=full_path, dataset="financial")
    # config = load_yaml(p / file_name)
    # return Path(config["MODEL_FOLDER"])


def train_ensemble(full_path: Path, dump_folder):
    file_name = "ensemble_test.yml"
    run(file_name, EnsembleBERT, full_path=full_path,
        split_fun=lambda n: ensemble_args_split(dump_folder, EPOCH, L2_MODEL, n), dataset="financial")


def train_supported_ensemble(full_path: Path, dump_folder):
    file_name = "support_test.yml"
    run(file_name, SupportedBERT, full_path=full_path,
        split_fun=lambda n: ensemble_args_split(dump_folder, EPOCH, None, n), dataset="financial")


def train_e2e(full_path: Path):
    file_name = "e2e_test.yml"
    run(file_name, E2EbiBERT, full_path=full_path,
        split_fun=lambda n: dict())


if __name__ == "__main__":
    # BASE_CONFIG_PATH = Path("src/models/hierarchical_labeling/configs")
    # print("Training L1 model")
    # l1_dump_folder = train_l1_model()  # Path("src/models/hierarchical_labeling/dumps/bert_l1")
    # train_l1_model()
    # train_l2_model()
    # l1_dump_folder = Path("src/models/hierarchical_labeling/dumps/bert_l1")
    # BASE_CONFIG_PATH = Path("src/models/hierarchical_labeling/configs/ensemble_tests")

    l1_dump_folder = Path("src/models/hierarchical_labeling/dumps/bert_financial_l1")
    BASE_CONFIG_PATH = Path("src/models/hierarchical_labeling/configs/ensemble_tests")

    # print("Training ensemble model")
    # train_ensemble(BASE_CONFIG_PATH, l1_dump_folder)

    print("Training supported model")
    train_supported_ensemble(BASE_CONFIG_PATH, l1_dump_folder)

    # print("Training end-2-end model")
    # train_e2e(BASE_CONFIG_PATH, )
