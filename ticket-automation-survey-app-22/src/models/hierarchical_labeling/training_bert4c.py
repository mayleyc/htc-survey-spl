from pathlib import Path
from typing import Callable

from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.models.hierarchical_labeling.modules.ensembles import EnsembleBERT, SupportedBERT, E2EbiBERT
from src.models.hierarchical_labeling.modules.utility_functions import run_training

# DEBUG PARAMETERS
workers = 0
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# RUN PARAMETERS
dataset_global = ["financial", "bugs"][0]
method = ["BERT4C", "ML-BERT", "SUPP-BERT", "DH-BERT"][0]
mode = ["validation", "test"][1]


def run(file_name, model_class, full_path: Path = None, split_fun: Callable = None, dataset=None):
    if not full_path:
        config_path = Path("src") / "models" / "hierarchical_labeling" / "configs" / file_name
    else:
        config_path = full_path / file_name
    if dataset:
        ds = dataset
    else:
        ds = dataset_global
    print(f"Mode: {mode}")
    print(f"Dataset: {ds}")

    run_training(config_path, ds, model_class, workers, validation=(mode == "validation"),
                 split_fun=split_fun)


if __name__ == "__main__":
    if method == "BERT4C":
        config_name = "various/bert_config_fin.yml"
        mod_cls = BERTForClassification
    elif method == "ML-BERT":
        config_name = "mllm_config.yml"
        mod_cls = EnsembleBERT
    elif method == "SUPP-BERT":
        config_name = "supplm_config.yml"
        mod_cls = SupportedBERT
    elif method == "DH-BERT":
        config_name = "dhlm_config.yml"
        mod_cls = E2EbiBERT
    else:
        raise ValueError(f"Unsupported method. Must be on of {method}.")

    run(config_name, mod_cls)
