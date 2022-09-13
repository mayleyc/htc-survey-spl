# Imports for SHAP MimicExplainer with LightGBM surrogate model
import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
# Split data into train and test
from sklearn.model_selection import RepeatedStratifiedKFold

from src.dataset_tools.data_preparation.prepare_linux_dataset import read_dataset as read_bugs
from src.models.multi_level.modules.utility_functions import _predict, _setup_training
from src.models.multi_level.multilevel_models import MultiLevelBERT, SupportedBERT
from src.training_scripts.multilevel_models.train_multilevel import ensemble_args_split, get_actual_dump_name
from src.utils.generic_functions import load_yaml


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes utilizing sklearn metrics

    :param y_true: true labels
    :param y_pred: predicted labels
    :param argmax_flag: Whether to apply an "argmax" function over predictions
    :return: metrics in a dictionary [metric_name]: value
    """
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="macro",
                                                                   zero_division="warn")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "accuracy": acc,
        "macro_f1": fscore,
        "macro_precision": precision,
        "macro_recall": recall,
    }
    report = classification_report(y_true, y_pred, output_dict=True)
    report.pop('accuracy')
    report.pop('macro avg')
    report.pop('weighted avg')
    return metrics, report


def main():
    df = read_bugs()
    config_path = "configs/error_analysis/multilevel_config_error.yml"
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
    results = []
    reports = []
    for train_index, test_index in splitter.split(tickets, labels):
        config = load_yaml(config_path)
        fold_i += 1
        x_train, x_test = tickets.iloc[train_index], tickets.iloc[test_index]
        y_train, y_test = labels_all.iloc[train_index], labels_all.iloc[test_index]

        # x_train = x_train[:2000]
        # x_test = x_test[:2000]
        # y_train = y_train[:2000]
        # y_test = y_test[:2000]

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

        trainer, _, test_loader = _setup_training(train_config=config, model_class=model_class,
                                                  workers=0,
                                                  data=x_train, labels=y_train,
                                                  data_val=x_test, labels_val=y_test)

        y_pred, y_true = _predict(trainer.model, test_loader)  # (samples, num_classes)

        y_pred = y_pred.argmax(axis=-1)

        metrics, report = compute_metrics(y_true, y_pred)
        reports.append(report)
        results.append(metrics)

        del trainer
        torch.cuda.empty_cache()
        # break

    # Average metrics over all folds and save them to csv
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    save_path: Path = Path("out/error_analysis")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(save_path / f"{model_class.__name__}.csv")

    # Reports
    df_conc = pd.concat([pd.DataFrame(x) for x in reports])
    df_means = df_conc.groupby(df_conc.index).mean()
    ordered = df_means.transpose().sort_values(by="f1-score", ascending=False)
    ordered["label"] = ordered.index
    labels = y_test.unique()
    ordered['label'] = ordered['label'].apply(lambda x: labels[int(x)])
    ordered.to_csv(save_path / f"{model_class.__name__}_errors.csv")
    print("DIO")


if __name__ == "__main__":
    main()
