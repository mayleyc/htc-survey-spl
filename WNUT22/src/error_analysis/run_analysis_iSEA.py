# Imports for SHAP MimicExplainer with LightGBM surrogate model
import re

import pandas as pd
# Split data into train and test
from sklearn.model_selection import RepeatedStratifiedKFold
import os
from src.dataset_tools.data_preparation.prepare_linux_dataset import read_dataset as read_bugs
from src.models.multi_level.modules.utility_functions import _predict, _setup_training
from src.models.multi_level.multilevel_models import MultiLevelBERT, SupportedBERT
from src.training_scripts.multilevel_models.train_multilevel import ensemble_args_split, get_actual_dump_name
from src.utils.generic_functions import load_yaml
import shap
import torch
import scipy as sp
import os
from pathlib import Path
from typing import Type
from src.utils.generic_functions import load_model_from_path
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer
from src.utils.losses import ce_loss
from src.dataset_tools.classes.torch_dataset import TransformerDatasetFlat
from src.dataset_tools.data_preparation.prepare_financial_dataset import read_dataset as read_financial
from src.dataset_tools.data_preparation.prepare_linux_dataset import read_dataset as read_bugs
from src.utils.generic_functions import dump_yaml, load_yaml
from src.utils.metrics import compute_metrics
from src.utils.torch_train_eval.early_stopper import EarlyStopping
from src.utils.torch_train_eval.grad_accum_trainer import GradientAccumulatorTrainer
from src.utils.torch_train_eval.trainer import Trainer
import numpy as np

from src.utils.generic_functions import extract_number, random_string


def load_previous(reload_path) -> Path:
    epoch_state_name = "epoch.json"
    model_checkpoint_name = "model.pt"
    """
    Load previously saved model. If the path passed is to a specific epoch that epoch will be loaded.
    If the path points to a dump folder with multiple epochs the last one will be loaded

    @param reload_path: path to model checkpoint, either to main folder or to specific epoch
    @param model_only: Only load weights and epoch state (not the optimizer)
    """
    # Folder of model
    reload_path = Path(reload_path)
    model_path = reload_path / model_checkpoint_name
    if reload_path.is_dir() and (not model_path.is_file()):
        # Passed path is the root folder of a dump, in this case the last epoch is loaded
        print("*** Loading the last epoch ... ***")
        epoch_folders_list = [x[0] for x in os.walk(reload_path)][1:]
        folder_to_load = max(epoch_folders_list, key=extract_number)
        folder_to_load = Path(folder_to_load)
        model_folder = reload_path
    else:
        # Else, the model path and opt path point to the specific epoch folder, so we load them directly
        model_folder = reload_path.parent
        folder_to_load = reload_path

    model_path = folder_to_load / model_checkpoint_name
    epoch_state_path = folder_to_load / epoch_state_name

    return model_path


def main():
    df = read_bugs()
    config_path = "configs/error_analysis/support_config_error.yml"
    seeds = load_yaml("configs/random_seeds.yml")
    config = load_yaml(config_path)

    fold_tot = config["NUM_FOLD"]
    repeats = config["CV_REPEAT"]

    tickets = df["message"]
    labels_all = labels = df[config["LABEL"]]
    if "ALL_LABELS" in config:
        labels_all: pd.DataFrame = df[config["ALL_LABELS"]]

    EPOCH = 2  # HARDCODED
    fold_i = 0

    splitter = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=repeats,
                                       random_state=seeds["stratified_fold_seed"])
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        x_train, x_test = tickets.iloc[train_index], tickets.iloc[test_index]
        y_train, y_test = labels_all.iloc[train_index], labels_all.iloc[test_index]

        x_train = x_train[:2000].reset_index(drop=True)
        x_test = x_test[:2000].reset_index(drop=True)
        y_train = y_train[:2000].reset_index(drop=True)
        y_test = y_test[:2000].reset_index(drop=True)

        # --------------------------------

        if "MODEL_L1" in config.keys():
            l1_dump = config['MODEL_L1']
            print(f"L1: {l1_dump}")
        if "MODEL_L2" in config.keys():
            l2_dump = config['MODEL_L2']
            print(f"L2: {l2_dump}")
        else:
            l2_dump = None  # for supported

        split_fun = lambda x: ensemble_args_split(l1_dump, l2_dump, EPOCH, x)
        config.update(split_fun(fold_i))
        config["PATH_TO_RELOAD"] = get_actual_dump_name(config["PATH_TO_RELOAD"], re.compile(f"fold_{fold_i}.*"))
        if config["mode"] == "Multilevel_ML":
            model_class = MultiLevelBERT
        elif config["mode"] == "Supported":
            model_class = SupportedBERT
        else:
            raise ValueError

        tokenizer = AutoTokenizer.from_pretrained(config["PRETRAINED_LM"])
        dataset_class = TransformerDatasetFlat

        train_data = dataset_class(x_train, y_train, remove_garbage=config["REMOVE_GARBAGE_TEXT"])
        test_data = dataset_class(x_test, y_test, remove_garbage=config["REMOVE_GARBAGE_TEXT"])

        config["n_class"] = train_data.n_y

        # model = model_class(**config)
        device = 'cuda'
        # model_folder = load_previous(reload_path=config["PATH_TO_RELOAD"])
        model = load_model_from_path(config["PATH_TO_RELOAD"])
        model.device = device
        model.model_1.device = device
        for sm in model.submodules():
            sm.device = device
        # for sm in model.model_1.submodules():
        #     sm.device = device
        if config["mode"] == "Multilevel_ML":
            model.model_2.device = device
        # x = [t for t in test_data.x]
        #
        # max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
        # encoded_x = tokenizer(x, truncation=True, max_length=max_len, padding=True)
        # item_x = {key: torch.tensor(val).to(device) for key, val in encoded_x.items()}
        # y = [t for t in test_data.y]
        # y = torch.LongTensor(y).to(device)
        #
        # model.train(False)
        # example = (item_x, y)
        # with torch.no_grad():
        #     y_pred, y_true = model(example, inference=True)
        #
        # y_pred_argmax = np.argmax(y_pred.detach().cpu().numpy(), axis=1).flatten()

        for genre in y_test.unique():
            # --------------------------------------------
            # Run the pretrained model and save model output
            # --------------------------------------------
            # print("test on", genre)
            indices = y_test[y_test == genre].index.to_list()
            # pred = y_pred_argmax[indices]
            # true = y_true[indices]
            # data_to_output = pd.DataFrame()
            # data_to_output['y_pred'] = pred
            # data_to_output['y_gt'] = true
            #
            # path = f"out/iSEA/{genre}/"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            #
            # data_to_output.to_csv(path + "model_output.csv", index=None)
            # # calculate accuracy
            # print("accuracy", (data_to_output['y_pred'] == data_to_output['y_gt']).astype(int).sum() / len(true))
            # --------------------------------------------
            # SHAP
            # --------------------------------------------
            # shap_f(pred_logits)
            explainer = shap.Explainer(shap_f, tokenizer)
            shap_values = explainer(x_test[indices])
            top_tokens = []
            for idx in range(shap_values.values.shape[0]):
                token_idx = []
                # 3 possible prediction values
                for pred_val in range(3):
                    temp_list = np.abs(shap_values.values[idx][:, pred_val])
                    order = np.argsort(temp_list)
                    '''save the top three tokens with the highest absolute SHAP value'''
                    largest_indices = order[::-1][:3]
                    token_idx.append([
                        {"token": shap_values.data[idx][largest_indices[0]],
                         'val': shap_values.values[idx][largest_indices[0]][pred_val]},
                        {"token": shap_values.data[idx][largest_indices[1]],
                         'val': shap_values.values[idx][largest_indices[1]][pred_val]},
                        {"token": shap_values.data[idx][largest_indices[2]],
                         'val': shap_values.values[idx][largest_indices[2]][pred_val]},
                    ])
                top_tokens.append(token_idx)
        #
        # FIXME: JUST FIRST FOLD
        break
    # input("Check it")


# SHAP


def shap_f(x):
    outputs = []
    for _x in x:
        encoding = torch.tensor([tokenizer.encode(_x)])
        output = model(encoding)[0].detach().cpu().numpy()
        outputs.append(output[0])
    outputs = np.array(outputs)
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    return val




if __name__ == "__main__":
    main()
    # example()
