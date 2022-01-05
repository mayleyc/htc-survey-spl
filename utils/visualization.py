import json
import math
import os
from collections import Counter
from typing import List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

"""
Visualization tools for datasets
"""


def labels_histogram(*label_columns: Union[List, pd.Series], multilabel: bool = True, top_n: int = None,
                     title: str = None) -> None:
    """
    Plot histogram of classes in the dataset. Argument is a list of pandas series, which are stacked.

    :param label_columns: series in which each element is a label (or list of)
    :param multilabel: whether the dataset is multilabel or not
    :param top_n: consider only the most frequent N labels
    :param title: title of the plot
    """
    n_columns: int = len(label_columns)
    if n_columns == 0:
        raise ValueError("At least one array required as input")
    elif n_columns == 1:
        label_column: pd.Series = label_columns[0]
    else:
        label_column: pd.Series = label_columns[0]
        label_columns = label_columns[1:]
        for column in label_columns:
            label_column = label_column.append(column)

    labels = label_column  # [literal_eval(x) if multilabel else x for x in label_column.values]
    all_labels = [label for lbs in labels for label in lbs] if multilabel else labels
    labels_count = Counter(all_labels).most_common(n=top_n)
    max_for_plot = 100
    for i in range(math.ceil(len(labels_count) / max_for_plot)):
        plt.figure(figsize=(15, 20), dpi=200)
        sns.set_style("darkgrid")
        offset = i * max_for_plot
        labs = labels_count[offset:(offset + max_for_plot)]
        y, x = zip(*labs)
        x = [*x, np.mean(x)]
        y = [*y, "MEAN"]
        ax = sns.barplot(x=pd.Series(x), y=pd.Series(y), log=True,
                         order=[e for _, e in reversed(sorted(zip(x, y), key=lambda rr: rr[0]))])

        # Add number over plot
        for p in ax.patches:
            tx = "%.d" % p.get_width()
            ax.annotate(tx, xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center")

        ax.set_title(title if title else "Labels with count (log scale)", fontsize=24)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()


def label_count_histogram(*all_labels: Iterable[Iterable[str]], title: str = None) -> None:
    """
    Barplot to visualize distribution of the number of labels

    :param all_labels: iterable containing the topics
    :param title: plot title
    """

    labs = [t for labels in all_labels for t in labels]

    c = Counter([len(t) for t in labs])
    data: List[Tuple[int, int]] = sorted(c.items(), reverse=True)
    label_number = [d for d, r in data]
    doc_count = np.array([r for d, r in data])

    plt.figure(figsize=(15, 15))
    plt.xlabel("Number of documents", fontsize=18)
    plt.ylabel("Label count", fontsize=18)
    sns.set_style("darkgrid")
    ax = sns.barplot(x=doc_count, y=label_number, log=True, orient="h", ci=None)

    # Add number over plot
    for p in ax.patches:
        tx = "%.d" % p.get_width()
        ax.annotate(tx, xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center")
    ax.set_title(title if title else "Number of labels per document", fontsize=24)
    plt.tight_layout()
    plt.show()


def extract_topics(plot: bool = False):
    """
    Parse the files extracted from dumps and produce barplot with the frequency ot the most frequent labels (100)
    """

    LANG = "it"
    TOP_K = 100

    topics: List[str] = list()
    for root, direc, files in os.walk("data/raw/{:s}wiki/extracted".format(LANG)):
        for file in files:
            path = os.path.join(root, file)
            with open(path) as f:
                line = f.readline()
                while line:
                    data = json.loads(line)
                    line = f.readline()

                    topics += data["topics"]

    labels_count = Counter(topics).most_common(TOP_K)
    a = [e[0] for e in labels_count]
    # Write top-K topics to file
    with open("data/raw/{:s}wiki/topics.yml".format(LANG), "w+") as rr:
        yaml.safe_dump(a, rr)
    if plot:
        labels_histogram(topics, multilabel=False, top_n=TOP_K,
                         title="Wikipedia {:s} dump top-{:d} frequent topics (log scale) [{:d}]".format(LANG.upper(), TOP_K, len(topics)))
    # multilabel=False to deal with already flattened list


if __name__ == "__main__":
    extract_topics()
