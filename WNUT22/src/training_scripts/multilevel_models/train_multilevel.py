import os
import re
from pathlib import Path
from typing import Dict
from typing import Optional

from src.models.multi_level.modules.utility_functions import run_training
from src.models.multi_level.multilevel_models import MultiLevelBERT, SupportedBERT
from src.utils.generic_functions import load_yaml

# DEBUG PARAMETERS
workers = 0


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


def ensemble_args_split(model_1: Path, model_2: Optional[Path], epoch: Optional[int], fold_n: int):
    """
    Idk la spieghi

    :param model_1: path to first model
    :param model_2: (optional) path to second model
    :param epoch: (optional) epoch to load. If none, loads latest
    :param fold_n: fold number
    :return: dictionary containing correct directories for models 2 load
    """
    m_1 = get_actual_dump_name(model_1, re.compile(f"fold_{fold_n}.*"))
    if epoch is None:
        # Prende le cartelle, prende l'ultima, prende il numero
        epochs = next(os.walk(m_1))[1]
        epoch = max([int(x.split("_")[1]) for x in epochs])

    model_1 = str(m_1 / f"epoch_{epoch}" / "model.pt")
    if model_2 is not None:
        m_2 = get_actual_dump_name(model_2, re.compile(f"fold_{fold_n}.*"))
        if epoch is None:
            # Prende le cartelle, prende l'ultima, prende il numero
            epochs = next(os.walk(m_2))[1]
            epoch = max([int(x.split("_")[1]) for x in epochs])
        model_2 = str(m_2 / f"epoch_{epoch}" / "model.pt")
        return dict(MODEL_L1=model_1, MODEL_L2=model_2)
    return dict(MODEL_L1=model_1)


def train_multilevel(model_class):
    config_base: Path = Path("configs") / "multi_level"
    out_base: Path = Path("out") / "multi_level"
    # --------------------------------
    # match model_class:
    if model_class == "MultiLevelBERT":
            mod_cls = MultiLevelBERT
            config_path = (config_base / "multilevel_config.yml")
    elif model_class == "SupportedBERT":
            mod_cls = SupportedBERT
            config_path = (config_base / "support_config.yml")

    config: Dict = load_yaml(config_path)
    # --------------------------------
    specific_model = f"{config['mode']}"
    out_folder = out_base / specific_model
    print(f"Specific model: {specific_model}")
    print(f"Dataset: {config['dataset']}")
    # --------------------------------
    if "MODEL_L1" in config.keys():
        l1_dump = config['MODEL_L1']
        print(f"L1: {l1_dump}")
    if "MODEL_L2" in config.keys():
        l2_dump = config['MODEL_L2']
        print(f"L2: {l2_dump}")
    else:
        l2_dump = None  # for supported
    # --------------------------------
    split_fun = lambda x: ensemble_args_split(l1_dump, l2_dump, EPOCH, x)
    run_training(config_path, config["dataset"], mod_cls, out_folder, workers,
                 validation=(config["validation"]), split_fun=split_fun)


if __name__ == "__main__":
    EPOCH = 3  # "None" to use last one
    train_multilevel("SupportedBERT")
    train_multilevel("MultiLevelBERT")
