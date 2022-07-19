import os
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset_tools.prepare_financial_dataset import read_dataset as read_financial
from src.dataset_tools.prepare_linux_dataset import read_dataset as read_bugs
from src.models.hierarchical_labeling.dataset import TransformerDatasetFlat, TransformerDataset2Levels
from src.models.hierarchical_labeling.modules.ensembles import E2EbiBERT
from src.models.hierarchical_labeling.modules.loss_fn import DoubleCE
from src.model_evaluation.metrics import compute_metrics
from src.utils.torch_train_eval.early_stopper import EarlyStopping
from src.utils.torch_train_eval.generic_functions import dump_yaml, load_yaml
from src.utils.torch_train_eval.trainer import Trainer


def collate_batch(tokenizer, batch, two_labels: bool = False):
    x = [t for t, *_ in batch]
    max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
    encoded_x = tokenizer(x, truncation=True, max_length=max_len, padding=True)
    item = {key: torch.tensor(val) for key, val in encoded_x.items()}
    if two_labels is True:
        y1 = [t for _, t, _ in batch]
        y2 = [t for _, _, t in batch]
        return item, torch.LongTensor(y1), torch.LongTensor(y2)
    else:
        y = [t for _, t in batch]
        return item, torch.LongTensor(y)


def compute_class_weights(ds) -> torch.FloatTensor:
    labs = ds.y.detach().cpu().numpy()
    labs = np.sort(labs)
    uniques, counts = np.unique(labs, return_counts=True)
    weights = compute_class_weight(class_weight="balanced", classes=uniques, y=labs)
    # tot = float(len(labs))
    # weights = counts / tot
    return torch.FloatTensor(weights)


def _setup_training(train_config, model_class: Type, workers: int, data, labels, data_val, labels_val):
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(train_config["PRETRAINED_LM"])
    two_levels_mode: bool = model_class == E2EbiBERT
    if two_levels_mode:
        dataset_class = TransformerDataset2Levels
        loss_class = DoubleCE
    else:
        dataset_class = TransformerDatasetFlat
        loss_class = CrossEntropyLoss
    train_data = dataset_class(data, labels, remove_garbage=train_config["REMOVE_GARBAGE_TEXT"])
    val_data = dataset_class(data_val, labels_val, remove_garbage=train_config["REMOVE_GARBAGE_TEXT"])
    train_config["n_class"] = train_data.n_y
    if two_levels_mode:
        train_config["n_class_l1"] = train_data.n_y1
    # Initialize model
    model = model_class(**train_config)

    # Initialize Optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=train_config["LEARNING_RATE"], weight_decay=train_config["L2_REG"])

    ce_args = dict()
    if train_config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
        ce_args = dict(weight=compute_class_weights(train_data).to(train_config["DEVICE"]))

    loss_func = loss_class(**ce_args)
    early_stopper = EarlyStopping(*train_config["EARLY_STOPPING"].values())
    # -------------------------------
    # Prepare dataset
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config["BATCH_SIZE"],
                                                  num_workers=workers, shuffle=True,
                                                  collate_fn=lambda x: collate_batch(tokenizer, x, two_levels_mode))
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, two_levels_mode))
    # -------------------------------
    # Initiate training
    trainer = Trainer(model, train_config, loss_func, opt, early_stopper, unpack_flag=False)
    return trainer, training_loader, validation_loader


def _predict(model, loader):
    model.train(False)
    y_pred = list()
    y_true = list()
    with torch.no_grad():
        for i, pred_data in tqdm(enumerate(loader), total=len(loader)):
            l, y, *_ = model(pred_data, inference=True)
            pred = torch.softmax(l, dim=1)
            y_pred.append(pred.detach().cpu().numpy())
            y_true.append(y.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)


def _training_testing_loop(config_path: Path, model_class: Type, workers: int, df: pd.DataFrame, validation: bool = False, save_name: str = None, split_fun=None):
    config = load_yaml(config_path)
    # General parameters
    fold_i = 0
    fold_tot = config["NUM_FOLD"]
    repeats = config["CV_REPEAT"]
    model_folder = Path(config["MODEL_FOLDER"])

    tickets = df["message"]
    labels_all = labels = df[config["LABEL"]]
    if "ALL_LABELS" in config:
        labels_all: pd.DataFrame = df[config["ALL_LABELS"]]

    # Start K-Fold CV, repeating it for better significance
    results = list()
    splitter = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=repeats, random_state=772361728)
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot * repeats} ({fold_tot} folds * {repeats} repeats)")
        config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")

        if split_fun is not None:
            config.update(split_fun(fold_i))
        if "MODEL_L1" in config.keys():
            print(f"L1: {config['MODEL_L1']}")
        if "MODEL_L2" in config.keys():
            print(f"L2: {config['MODEL_L2']}")
        x_train, x_test = tickets.iloc[train_index], tickets.iloc[test_index]
        y_train, y_test = labels_all.iloc[train_index], labels_all.iloc[test_index]
        if validation is True:
            # Replace test set with validation set. Notice that the test set will be ignored, and never used in validation or training
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=936818217,
                                                                stratify=labels.iloc[train_index])

        # FOR DEBUG
        # x_train = x_train[:100]
        # x_test = x_test[:100]
        # y_train = y_train[:100]
        # y_test = y_test[:100]

        # Create and train a model
        trainer, train_load, val_load = _setup_training(train_config=config, model_class=model_class,
                                                        workers=workers,
                                                        data=x_train, labels=y_train,
                                                        data_val=x_test, labels_val=y_test)
        # TODO: Change me if you need to validate
        # trainer.train(train_load, val_load if validation else None)
        trainer.train(train_load, val_load)

        # TEST the model
        model = trainer.model
        # Use the model to predict test/validation samples
        y_pred, y_true = _predict(model, val_load)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred)
        # Save metric for current fold
        results.append(metrics)

        # Necessary for sequential run. Empty cache should be automatic, but best be sure.
        del trainer, model
        torch.cuda.empty_cache()

    # Average metrics over all folds and save them to csv
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    save_name = save_name if save_name is not None else f"results_{'val' if validation else 'test'}"
    save_path = model_folder / "results"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(save_path / (save_name + ".csv"))
    dump_yaml(config, save_path / (save_name + ".yml"))


def run_training(config_path: Path, dataset: str, model_class, workers: int, validation, split_fun=None):
    if dataset == "financial":
        df = read_financial()
    else:
        df = read_bugs()
    _training_testing_loop(config_path, model_class, workers, df, validation, split_fun=split_fun)
