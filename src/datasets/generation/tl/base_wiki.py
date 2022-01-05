import json
import os
from operator import itemgetter
from typing import List, Set, Dict

import joblib
import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from src.datasets.generation.base import BaseDataset
from utils.general import load_yaml
from utils.split import stratified_split
from utils.visualization import labels_histogram, label_count_histogram


class BaseWiki(BaseDataset):
    """
    Generation utility for the Wikipedia dataset. Manage the following tasks
      - read the raw dataset extracted with WikiExtractor
      - filter articles with categories defined in topics.yml
      - downsample the most frequent classes, both to balance and to reduce data samples
      - split into training, test and evaluation sets, and save them to files in FastText format
    """
    MULTILABEL: bool = True

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.__labels: Dict[int, List[str]] = dict()  # art_id: labels
        self.__texts: Dict[int, str] = dict()  # art_id: text

    def generate(self, plot: bool = False) -> None:
        """
        Start generation of Wikipedia dataset splits and dump it to file

        :param plot: whether to display plots
        """
        print("Reading raw dataset...")
        chosen_topics: Set[str] = set(load_yaml(self.RAW_ROOT / "topics.yml"))
        line_count = 0
        for root, direc, files in os.walk(self.RAW_ROOT / "extracted"):
            for file in files:
                path = os.path.join(root, file)
                with open(path) as f:
                    line = f.readline()
                    while line:
                        data = json.loads(line)
                        line = f.readline()

                        topics = [t for t in data["topics"] if t in chosen_topics]
                        if not topics:
                            continue

                        self.__labels[line_count] = topics
                        self.__texts[line_count] = " ".join(data["text"].split())
                        line_count += 1

        if plot:
            print("Plot labels frequency...")
            lbs = list(self.__labels.values())
            labels_histogram(lbs, multilabel=True, title=f"{self.__class__.__name__} topic frequency (log scale)")
            label_count_histogram(lbs, title=f"{self.__class__.__name__} number of topics per document")

        # DOWNSAMPLE DATASET FREQUENT CATEGORIES

        print("Downsampling articles...")
        label_to_article: Dict[str, List[int]] = dict()  # map labels -> article ids
        for i in tqdm.tqdm(sorted(self.__texts.keys())):
            ls = self.__labels[i]
            for lab in ls:
                if lab in label_to_article:
                    label_to_article[lab].append(i)
                else:
                    label_to_article[lab] = [i]

        # remove articles from categories with more than {max_num} articles
        max_num = 50000
        for _, articles in tqdm.tqdm(label_to_article.items()):
            # order articles by number of categories, and by length of text, filtering the ones still present
            articles = sorted([(len(self.__labels[a]), len(self.__texts[a]), a) for a in articles if a in self.__texts],
                              key=itemgetter(0, 1), reverse=True)

            to_remove = articles[max_num:]
            if to_remove:
                for _, _, rem in to_remove:
                    self.__labels.pop(rem)
                    self.__texts.pop(rem)

        # STRATIFY SPLITS

        print("Encoding labels...")
        encoder = MultiLabelBinarizer()
        target = encoder.fit_transform(list(self.__labels.values()))
        joblib.dump(encoder, self.ENCODER_FILEPATH)

        print("Splitting training and test set...")
        x_train, x_test, y_train, y_test = stratified_split(list(self.__texts.values()), target,
                                                            splits=(self.TRAIN_SIZE, 1 - self.TRAIN_SIZE))
        print("Splitting training and validation set...")
        x_train, x_val, y_train, y_val = stratified_split(x_train, y_train, splits=(1 - self.VAL_SIZE, self.VAL_SIZE))

        y_train = encoder.inverse_transform(y_train)
        y_test = encoder.inverse_transform(y_test)
        y_val = encoder.inverse_transform(y_val)

        print(f"\n++++++++++++++++++++++++++++++++++\nTotal samples: {len(self.__texts)}")
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Evaluation samples: {len(x_val)}\n++++++++++++++++++++++++++++++++++\n")
        assert len(self.__texts) == len(x_train) + len(x_test) + len(x_val)

        if plot:
            print("Plotting splits statistics...")
            labels_histogram(y_train, title=f"{self.__class__.__name__} training topics (log scale)")
            labels_histogram(y_test, title=f"{self.__class__.__name__} test topics (log scale)")
            labels_histogram(y_val, title=f"{self.__class__.__name__} validation topics (log scale)")

        # SAVE SPLITS WITH TXT FORMAT

        print("Saving data...")
        self._save_to_txt(x_train, x_test, x_val, y_train, y_test, y_val)
