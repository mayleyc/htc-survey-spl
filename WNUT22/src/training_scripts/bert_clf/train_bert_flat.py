from pathlib import Path
from typing import Dict

from src.models.bert_clf.bert_classifier import BERTForClassification
from src.models.multi_level.modules.utility_functions import run_training
from src.utils.generic_functions import load_yaml

# DEBUG PARAMETERS
workers = 0


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_bert4c():
    config_base: Path = Path("configs") / "bert"  # / "lr_validation"
    out_base: Path = Path("out") / "bert4c"
    configs = ["bert_l1_config.yml", "bert_l2_config.yml"]

    for c in configs:
        config_path = (config_base / c)
        config: Dict = load_yaml(config_path)
        mod_cls = BERTForClassification
        specific_model = f"{config['mode']}_{config['CLF_STRATEGY']}_{config['CLF_STRATEGY_NUM_LAYERS']}"
        out_folder = out_base / specific_model
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")

        run_training(config_path, config["dataset"], mod_cls, out_folder, workers,
                     validation=(config["validation"]))


if __name__ == "__main__":
    train_bert4c()
