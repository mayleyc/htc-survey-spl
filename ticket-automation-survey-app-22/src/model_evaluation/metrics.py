from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, argmax_flag: bool = True) -> Dict[str, float]:
    """
    Computes utilizing sklearn metrics

    :param y_true: true labels
    :param y_pred: predicted labels
    :param argmax_flag: Whether to apply an "argmax" function over predictions
    :return: metrics in a dictionary [metric_name]: value
    """
    # Compute metrics with sklearn
    if argmax_flag:
        y_pred = y_pred.argmax(axis=-1)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="macro",
                                                                   zero_division="warn")
    w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                         average="weighted",
                                                                         zero_division="warn")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    w_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "accuracy": acc,
        "macro_f1": fscore,
        "macro_precision": precision,
        "macro_recall": recall,
        "w_accuracy": w_acc,
        "w_f1": w_fscore,
        "w_precision": w_precision,
        "w_recall": w_recall
    }
    return metrics
