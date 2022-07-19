import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
# from keras.saving.save import load_model
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from src.model_evaluation.metrics import compute_metrics
from src.models.deep_triage.data_preparation import prepare_train_test_data, prepare_for_model
from src.models.deep_triage.modules.deeptriage import DeepTriage
from src.utils.torch_train_eval.generic_functions import load_yaml, dump_yaml

warnings.filterwarnings('ignore')


def compute_class_weights(labs):
    labs = np.sort(labs)
    uniques, counts = np.unique(labs, return_counts=True)
    weights = compute_class_weight(class_weight="balanced", classes=uniques, y=labs)
    weights = dict(zip(range(len(uniques)), weights))
    # tot = float(len(labs))
    # weights = counts / tot
    return weights


def training_testing_loop(tickets: List[List[str]], labels: List[str], w2v, config_path: Path, early_stopping,
                          checkpoint_callback, validation: bool = False,
                          save_name: str = None):
    # This is ugly but I blame someone else
    config = load_yaml(config_path)
    # General parameters
    fold_i = 0
    fold_tot = config["numCV"]
    bs = config["batch_size"]
    epoch_n = config["epochs"]
    test_bs = 2 * bs
    classes = list(set(labels))

    # Create model and print summary
    deep_triage = DeepTriage(config, num_classes=len(classes), vocab_size=len(w2v.wv.key_to_index),
                             embeddings=w2v.wv.vectors)
    deep_triage.summary()

    tickets = np.array(tickets)
    labels = np.array(labels)

    # Start K-Fold CV, repeating it for better significance
    results = list()
    repeats = config["CV_REPEAT"]
    splitter = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=config["CV_REPEAT"], random_state=772361728)
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot * repeats} ({fold_tot} folds * {repeats} repeats)")
        # if config["dataset"] == "financial" and fold_i != 3:
        #     print("Skipperooo!")
        #     continue
        # elif config["dataset"] == "bugs" and fold_i <= 3:
        #     print("Skipperooo!")
        #     continue
        x_train, x_test = tickets[train_index], tickets[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if validation is True:
            # Replace test set with validation set. Notice that the test set will be ignored, and never used in validation or training
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=936818217,
                                                                stratify=y_train)

        fit_args = dict()
        if config["CLASS_BALANCED_WEIGHTED_LOSS"] is True:
            class_w = compute_class_weights(y_train)
            fit_args = dict(class_weight=class_w)

        # Prepare data for model
        x_train, y_train = prepare_for_model(x_train, y_train, classes, w2v, config, embeddings=False)
        x_test, y_test = prepare_for_model(x_test, y_test, classes, w2v, config, embeddings=False)

        # Create and train a model
        checkpoint = config["LOAD_MODEL"]
        if checkpoint:
            deep_triage = DeepTriage(config, num_classes=len(classes), vocab_size=len(w2v.wv.key_to_index),
                                     embeddings=w2v.wv.vectors)
            deep_triage.model = load_model(checkpoint, compile=True)
        else:
            deep_triage = DeepTriage(config, num_classes=len(classes), vocab_size=len(w2v.wv.key_to_index),
                                     embeddings=w2v.wv.vectors)

        # --- Compile model ---
        deep_triage.compile()
        # ---------------------
        # if validation is True:
        hist = deep_triage.model.fit(x_train, y_train, batch_size=bs, epochs=epoch_n,
                                     validation_batch_size=test_bs,
                                     validation_data=(x_test, y_test),
                                     callbacks=[early_stopping, checkpoint_callback], **fit_args)
        # else:
        #     hist = deep_triage.model.fit(x_train, y_train, batch_size=bs, epochs=epoch_n, **fit_args)

        # Use the model to predict test/validation samples
        y_pred = deep_triage.model.predict(x_test, batch_size=test_bs)  # (samples, num_classes)

        # Compute metrics with sklearn
        y_true = y_test.argmax(axis=-1)
        y_pred = y_pred.argmax(axis=-1)
        metrics = compute_metrics(y_true, y_pred, False)

        # Save metric for current fold
        results.append(metrics)

    # Average metrics over all folds and save them to csv
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    save_name = save_name if save_name is not None else f"results_{'val' if validation else 'test'}"
    folder = config_path.parent.parent / f"results_{config['dataset']}"
    os.makedirs(folder, exist_ok=True)
    df.to_csv(folder / (save_name + ".csv"))
    dump_yaml(config, folder / (save_name + ".yml"))


# def tuning_model():
#     lrs = [1e-4, 1e-5]
#     max_sentence_lens = [100, 500]
#     i = 0
#     for max_sentence_len in max_sentence_lens:
#         i += 1
#         print(f"Testing {max_sentence_len} of sequence length")
#         config["max_sentence_len"] = max_sentence_len
#         # training_testing_loop(data, y1, validation=True, save_name=f"results_l1_{i}", config=config)
#         # training_testing_loop(data, y2, validation=True, save_name=f"results_l2_{i}", config=config)
#         training_testing_loop(data, yf, validation=True, save_name=f"results_flat_{i}", config=config)

def run(config_path, ):
    config = load_yaml(config_path)
    dataset = config["dataset"]
    w2v, data, _, _, yf = prepare_train_test_data(dataset=dataset, config=config)

    early_stopping = EarlyStopping(monitor="val_loss", patience=2)
    checkpoint_callback = ModelCheckpoint(config_path.parent / "dumps" / f"{config['dataset']}" / "deep_triage.h5",
                                          monitor="val_loss", verbose=1, save_best_only=True,
                                          mode="min", save_weights_only=False)

    print("Starting training and testing with CV ...")
    training_testing_loop(data, yf, w2v, config_path=config_path, early_stopping=early_stopping,
                          checkpoint_callback=checkpoint_callback, validation=config["validation"])


if __name__ == "__main__":
    config_path = Path("src") / "models" / "deep_triage" / "configs" / "config_financial.yml"
    run(config_path)

    config_path = Path("src") / "models" / "deep_triage" / "configs" / "config_bugs.yml"
    run(config_path)
