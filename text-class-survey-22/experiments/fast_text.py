import logging
import os
import re
from typing import List, Dict

import fasttext
import joblib
import pandas as pd
import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from src.datasets.generation.base import BaseDataset
from utils.general import load_yaml

dataset = BaseDataset


def main(run_conf: Dict):
    ds_name = run_conf["DATASET_NAME"]
    model_dump = os.path.join(run_conf["MODELS_PATH"], "FT_" + ds_name, run_conf["CHECKPOINT_NAME"])

    os.makedirs(os.path.dirname(model_dump), exist_ok=True)

    if run_conf["RELOAD"]:
        model = fasttext.load_model(model_dump)

    if run_conf["TRAIN"]:
        model = fasttext.train_supervised(input=str(dataset.generated_root(ds_name) / "train.txt"),
                                          autotuneValidationFile=str(dataset.generated_root(ds_name) / "val.txt"),
                                          autotuneDuration=run_conf["TUNING_TIME"],
                                          # autotuneModelSize="500M",
                                          loss="ova")
        model.save_model(model_dump)

        n, precision, recall = model.test(str(dataset.generated_root(ds_name) / "val.txt"), k=-1, threshold=0.5)
        logging.info(
            f"Validation set samples: {n}\nValidation precision: {precision:.2f}\nValidation recall: {recall:.2f}")

    if run_conf["TEST"]:
        logging.info(f"\n\n\n ***************** {ds_name} *****************")
        logging.info("Testing model with FT test function...")
        n, precision, recall = model.test(str(dataset.generated_root(ds_name) / "test.txt"), k=-1, threshold=0.5)
        logging.info(f"\nTest set samples: {n}\nTest precision: {precision:.2f}\nTest recall: {recall:.2f}")

        logging.info("Testing model with custom metrics...")
        decoder: LabelBinarizer = joblib.load(dataset.generated_root(ds_name) / "encoder.bin.xz")
        predictions: List[List[str]] = list()
        g_truth: List[List[str]] = list()
        # Get prediction for every text in test set
        with open(str(dataset.generated_root(ds_name) / "test.txt"), "r") as test_set:
            for line in tqdm.tqdm(test_set.readlines()):
                labels = re.findall(r"__label__(\w+)", line)
                text = line[len(" ".join(labels)) + 1:].strip("\n")

                pred_labels: List[str] = model.predict(text, k=-1, threshold=.5)[0]
                pred_labels = [cat[len("__label__"):] for cat in pred_labels]
                predictions.append(pred_labels)
                g_truth.append(labels)

        # Encode predictions and labels
        y_pred = decoder.transform(predictions)
        y_true = decoder.transform(g_truth)

        pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        metrics = dict(accuracy=acc, precision=pre, recall=rec, f1_score=f1)

        pp = pd.Series(metrics).to_string(dtype=False)
        logging.info(f"\n*** FT test metrics ***\n{pp}")


if __name__ == "__main__":
    conf = load_yaml(os.path.join("configs", "ft_run.yml"))
    logging.basicConfig(level=logging.INFO)
    main(conf)
