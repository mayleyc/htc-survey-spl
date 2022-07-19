from pathlib import Path

from src.models.hierarchical_labeling.ensemble_tests import ensemble_args_split
from src.models.hierarchical_labeling.modules.ensembles import EnsembleBERT, SupportedBERT, E2EbiBERT
from src.models.hierarchical_labeling.training_bert4c import run

L2_MODEL = Path("src/models/hierarchical_labeling/dumps/bert_test_24")
L1_MODEL = Path("src/models/hierarchical_labeling/dumps/bert_l1")
EPOCH = 3


# def train_ensemble(f_name: str):
#     model_cls = EnsembleBERT
#     main(f_name, model_cls, split_fun=lambda n: ensemble_args_split(L1_MODEL, EPOCH, L2_MODEL, n))


def train_supported_ensemble(f_name: str):
    run(f_name, SupportedBERT, split_fun=lambda n: ensemble_args_split(L1_MODEL, EPOCH, None, n))


def train_e2e(f_name: str):
    run(f_name, E2EbiBERT, split_fun=lambda n: dict())


if __name__ == "__main__":
    supp_tests = [f"val_{i}_ens.yml" for i in range(8, 11)]
    e2e_tests = [f"val_{i}_ens.yml" for i in range(11, 15)]
    for name in supp_tests:
        train_supported_ensemble(name)
    for name in e2e_tests:
        train_e2e(name)
