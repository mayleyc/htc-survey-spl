#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import multiprocessing
from pathlib import Path
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
from fasttext import train_supervised
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.dataset_tools.prepare_financial_dataset import read_dataset as read_financial
from src.dataset_tools.prepare_linux_dataset import read_dataset as read_bugs
from src.model_evaluation.metrics import compute_metrics
from src.utils.text_utilities.preprocess import text_cleanup
from src.utils.torch_train_eval.generic_functions import load_yaml, dump_yaml

FT_PARAMS = {
    "minCount": 1,
    "minCountLabel": 0,
    "wordNgrams": 1,
    "bucket": 2000000,
    "minn": 0,
    "maxn": 0,
    "t": 1e-4,
    "label": "__label__",
    "lr": 0.1,
    "lrUpdateRate": 100,
    "dim": 300,
    "ws": 5,
    "epoch": 5,
    "neg": 5,
    "thread": multiprocessing.cpu_count() - 1,
    "pretrainedVectors": '',
    "verbose": 2,
    "seed": 0,
    "loss": "softmax"
}


def find_metrics(model, test_data):
    predictions = list()
    g_truth = list()
    # Get prediction for every text in test set
    with open(test_data, "r", encoding="utf8") as test_set:
        for line in test_set.readlines():
            label, body = line.split(' ', 1)

            pred_label: str = model.predict(body.strip())[0][0]
            predictions.append(pred_label)
            g_truth.append(label)

    y_pred = np.array(predictions)
    y_true = np.array(g_truth)

    metrics = compute_metrics(y_true, y_pred, argmax_flag=False)

    return metrics


def save_to_fasttext_format(x: pd.Series, y: pd.Series, filename: Path):
    # forcing to clean file
    with open(filename, "w") as reset:
        reset.close()

    with open(filename, "a+", encoding="utf8") as f:
        for i in range(0, y.shape[0]):
            f.write(f"__label__{y.iloc[i]} {x.iloc[i].strip()}\n")


def get_model_parameters(model):
    args_getter = model.f.getArgs()

    parameters = {}
    for param in FT_PARAMS:
        attr = getattr(args_getter, param)
        if param == "loss":
            attr = attr.name
        parameters[param] = attr

    return parameters


# ---------------------------------------
MODEL_FOLDER: Path = Path("src") / "models" / "ticket_tagger"
CONFIG: Path = MODEL_FOLDER / "config.yml"
RUN_FOLDER: Path = MODEL_FOLDER / "dumps"
# ---------------------------------------
train_data_name: str = "train.txt"
train_data_val_name: str = "train_val.txt"
val_data_name: str = "val.txt"
test_data_name: str = "test.txt"
model_data_name: str = "model.bin"
best_params: str = "params.json"


def run_training(df: pd.DataFrame, config: Dict):
    tickets = text_cleanup(df["message"])
    labels = df[config["LABEL"]]
    # Create folder for models / splits
    RUN_FOLDER.mkdir(exist_ok=True, parents=True)
    # ---------------------------------------
    fold_tot = config["NUM_FOLD"]
    repeats = config["CV_REPEAT"]
    pre_trained_vecs = config["PRE_TRAINED_VECTORS"]
    emb_args = dict(pretrainedVectors=pre_trained_vecs, dim=300) if \
        Path(pre_trained_vecs).exists() and \
        Path(pre_trained_vecs).is_file() else dict()
    # ---------------------------------------
    print(f"Starting training and testing with CV (k={fold_tot})...")
    fold_i: int = 0
    # all_metrics: Dict = dict(accuracies=[], precisions=[], recalls=[], f1_scores=[])
    splitter: RepeatedStratifiedKFold = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=repeats,
                                                                random_state=772361728)
    results = list()
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot}")
        x_train, x_test = tickets.iloc[train_index], tickets.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        # ---------------------------------------
        # Divide in folders to keep data
        fold_folder: Path = RUN_FOLDER / f"fold_{fold_i}"
        fold_folder.mkdir(exist_ok=True)
        train_data: Path = fold_folder / train_data_name  # Training data
        train_data_val: Path = fold_folder / train_data_val_name  # Training set without validation data
        val_data: Path = fold_folder / val_data_name  # Validation data
        test_data: Path = fold_folder / test_data_name  # Test set data
        model_data: Path = fold_folder / model_data_name
        output_parameters_path: Path = fold_folder / best_params
        # Dump the current fold in FT format
        save_to_fasttext_format(x_train, y_train, train_data)
        save_to_fasttext_format(x_test, y_test, test_data)
        # ---------------------------------------
        if config["autoTune"]:
            # Notice that the test set will be ignored, and never used in validation or training
            # Test results will be done after
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=936818217,
                                                              stratify=y_train)
            # Overwrites train data to remove validation data
            save_to_fasttext_format(x_train, y_train, train_data_val)
            save_to_fasttext_format(x_val, y_val, val_data)

            model = train_supervised(
                input=str(train_data_val),
                autotuneValidationFile=str(val_data),
                autotuneDuration=config["autoTuneDuration"],
                **emb_args
            )

            # Save best hyperparameters
            best_hyperparams = get_model_parameters(model)
            pprint(best_hyperparams)
            with open(output_parameters_path, "w") as f:
                json.dump(best_hyperparams, f)

            model = train_supervised(input=str(train_data),
                                     **best_hyperparams)
        # ---------------------------------------
        else:
            model = train_supervised(input=str(train_data), **emb_args)
        # ---------------------------------------
        model.save_model(str(model_data))
        # Find metrics on TEST set for the resulting model OF THIS FOLD
        metrics = find_metrics(model, str(test_data))
        results.append(metrics)
    # ---------------------------------------
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    results_folder: Path = (RUN_FOLDER / "results")
    results_folder.mkdir(exist_ok=True)
    df.to_csv(results_folder / f"test_results.csv")
    dump_yaml(config, results_folder / f"test_config.yml")


# ----------------------------------------------------------------------

def main(data):
    config = load_yaml(CONFIG)
    global RUN_FOLDER
    RUN_FOLDER = RUN_FOLDER / data

    if data == "financial":
        df = read_financial()
    else:
        df = read_bugs()

    run_training(df, config)


if __name__ == "__main__":
    for ds in ["financial", "bugs"]:
        main(ds)
