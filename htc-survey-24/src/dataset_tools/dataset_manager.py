import itertools
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Callable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from src.dataset_tools import *
from src.utils.generic_functions import load_yaml


class DatasetManager:
    SEEDS_PATH: Path = Path("config/random_seeds.yml")

    def __init__(self, dataset_name: str, training_config: Dict):
        objective: str = training_config.get("objective", "multilabel")
        if objective == "multilabel":
            self.binarizer = MultiLabelBinarizer(sparse_output=False)
        elif objective == "multiclass":
            self.binarizer = None
        else:
            raise ValueError("Invalid target option. Choices: [multilabel, multiclass]")
        self.objective: str = objective
        self.seeds = load_yaml(self.SEEDS_PATH)
        # ---
        self.train_data, self.train_labels = None, None
        self.test_data, self.test_labels = None, None
        # ---
        self.splitter = RepeatedStratifiedKFold(n_splits=training_config["stratifiedCV"],
                                                n_repeats=training_config["n_repeats"],
                                                random_state=self.seeds["stratified_fold_seed"])
        self.dataset_name = dataset_name
        
        # --- Linux Bugs ---
        if dataset_name == "bugs":
            self.__load_unsplit(read_bugs)
        # --- Web of Science ---
        elif dataset_name == "wos":
            self.ohe_csv = wos_ohe_csv
            self.ohe = pd.read_csv(self.ohe_csv)
            self.__load_unsplit(read_wos_dataset)
            
        # --- Amazon ---
        elif dataset_name == "amazon":
            self.ohe_csv = amz_ohe_csv
            self.ohe = pd.read_csv(self.ohe_csv)
            self.__load_unsplit(get_amazon)
        # --- Blurb Genre Collection ---
        elif dataset_name == "bgc":
            self.ohe_csv = bgc_ohe_csv
            self.ohe = pd.read_csv(self.ohe_csv)
            self.__load_blurb()
        # --- RCV1-v2 ---
        elif dataset_name == "rcv1":
            self.__load_rcv1()
        else:
            raise ValueError("'dataset_name' invalid. Choose between: [bugs, wos, bgc, rcv1, amazon].")

    def __load_unsplit_old(self, read_function: Callable) -> None:
        # Load from file
        samples_list = read_function()
        # Stratified split does not work on multilabel. We split on deepest category (these are depth 2 datasets)
        self.labels_for_splitter = [doc['labels'][-1] for doc in samples_list]
        # Extract labels
        if self.objective == "multilabel":
            all_labels: List = [doc['labels'] for doc in samples_list]
            # Fit the binarizer on all labels
            all_labels: np.ndarray = self.binarizer.fit_transform(all_labels)
        elif self.objective == "multiclass":
            all_labels = self.labels_for_splitter
            # all_labels: np.ndarray = self.binarizer.fit_transform(self.labels_for_splitter)
            pass
        else:
            raise Exception("This should never be reached")
        # Set overall length of dataset
        self.len: int = len(all_labels)
        self.labels: np.ndarray = all_labels
        self.data: List[str] = [doc['text'] for doc in samples_list]

    def __load_unsplit(self, read_function: Callable) -> None:
        # Load from file
        samples_list = read_function()
        # Stratified split does not work on multilabel. We split on deepest category (these are depth 2 datasets)
        self.labels_for_splitter = [doc['labels'][-1] for doc in samples_list]
        all_labels_list = list()
        # Extract labels
        if self.objective == "multilabel":
            for label in self.labels_for_splitter:
                matches = self.ohe[self.ohe.iloc[:, 0] == label]
                one_hot_vectors = matches.iloc[:, 1:].to_numpy() #remove first column
                if one_hot_vectors.size > 0:
                    all_labels_list.append(one_hot_vectors)  # Note: append, NOT extend
                else:
                    raise ValueError(f"Label '{label}' not found in one-hot encoding CSV.")
            
            all_labels = np.vstack(all_labels_list) # ohe matrix

        elif self.objective == "multiclass":
            all_labels = self.labels_for_splitter
            # all_labels: np.ndarray = self.binarizer.fit_transform(self.labels_for_splitter)
            pass
        else:
            raise Exception("This should never be reached")
        # Set overall length of dataset
        self.len: int = len(all_labels)
        self.labels: np.ndarray = all_labels
        self.data: List[str] = [doc['text'] for doc in samples_list]

    def __load_blurb_old(self) -> None:
        # Load from file
        samples_train_list = get_bgc_split_jsonl("train")
        samples_test_list = get_bgc_split_jsonl("test")
        samples_val_list = get_bgc_split_jsonl("val")
        # Extract all labels
        if self.objective == "multilabel":
            train_labels: List = [doc['labels'] for doc in samples_train_list]
            test_labels: List = [doc['labels'] for doc in samples_test_list]
            val_labels: List = [doc['labels'] for doc in samples_val_list]
        elif self.objective == "multiclass":
            train_labels: List = [doc['labels'][-1] for doc in samples_train_list]
            test_labels: List = [doc['labels'][-1] for doc in samples_test_list]
            val_labels: List = [doc['labels'][-1] for doc in samples_val_list]
        else:
            raise Exception("This should never be reached")
        # Concatenate to fit binarizer
        all_labels = list(itertools.chain(train_labels, test_labels, val_labels))
        # Set overall length of dataset
        self.len: int = len(all_labels)
        if self.objective == "multilabel":
            # Fit the binarizer on all labels
            self.binarizer.fit(all_labels)
            # Binarize labels
            self.train_labels: np.ndarray = self.binarizer.transform(train_labels)
            self.test_labels: np.ndarray = self.binarizer.transform(test_labels)
            self.val_labels: np.ndarray = self.binarizer.transform(val_labels)
        else:
            self.train_labels = train_labels
            self.test_labels = test_labels
            self.val_labels = val_labels
        # Fetch text
        self.train_data: List[str] = [doc['text'] for doc in samples_train_list]
        self.test_data: List[str] = [doc['text'] for doc in samples_test_list]
        self.val_data: List[str] = [doc['text'] for doc in samples_val_list]
        
    def __load_blurb(self) -> None:
        # Load from file
        samples_train_list = get_bgc_split_jsonl("train")
        samples_test_list = get_bgc_split_jsonl("test")
        samples_val_list = get_bgc_split_jsonl("val")
        # Extract all labels
        if self.objective == "multilabel" or self.objective == "multiclass":
            train_labels: List = [doc['labels'][-1] for doc in samples_train_list]
            test_labels: List = [doc['labels'][-1] for doc in samples_test_list]
            val_labels: List = [doc['labels'][-1] for doc in samples_val_list]
            if self.objective == "multilabel":
                train_labels_multi: List = [doc['labels'] for doc in samples_train_list]
                test_labels_multi: List = [doc['labels'] for doc in samples_test_list]
                val_labels_multi: List = [doc['labels'] for doc in samples_val_list]
        else:
            raise Exception("This should never be reached")
        # Concatenate to fit binarizer                                               
        all_labels_multi = list(itertools.chain(train_labels_multi, test_labels_multi, val_labels_multi))
        # Set overall length of dataset
        self.len: int = len(all_labels_multi)
        if self.objective == "multilabel":
            # Fit the binarizer on all labels
            label_sets = [train_labels, val_labels, test_labels]
            for i, labels in enumerate(label_sets):
                all_labels_list = list()
                for label in labels:
                    matches = self.ohe[self.ohe.iloc[:, 0] == label]
                    one_hot_vectors = matches.iloc[:, 1:].to_numpy() #remove first column
                    if one_hot_vectors.size > 0:
                        all_labels_list.append(one_hot_vectors)  # Note: append, NOT extend
                    else:
                        raise ValueError(f"Label '{label}' not found in one-hot encoding CSV.")
                
                label_sets[i] = np.vstack(all_labels_list) # ohe matrix 
            # Update original variables with one-hot encoded arrays
            train_labels, val_labels, test_labels = label_sets
        else:
            self.train_labels = train_labels
            self.test_labels = test_labels
            self.val_labels = val_labels
        # Fetch text
        self.train_data: List[str] = [doc['text'] for doc in samples_train_list]
        self.test_data: List[str] = [doc['text'] for doc in samples_test_list]
        self.val_data: List[str] = [doc['text'] for doc in samples_val_list]

    def __load_rcv1(self):
        # Load from file
        samples_train_list = get_rcv1_split("train")
        samples_test_list = get_rcv1_split("test")
        self.labels_for_splitter = [doc['labels'][-1] for doc in samples_train_list]
        # Extract all labels
        if self.objective == "multilabel":
            train_labels: List = [doc['labels'] for doc in samples_train_list]
            test_labels: List = [doc['labels'] for doc in samples_test_list]
        elif self.objective == "multiclass":
            train_labels: List = [doc['labels'][-1] for doc in samples_train_list]
            test_labels: List = [doc['labels'][-1] for doc in samples_test_list]
        else:
            raise Exception("This should never be reached")
        # Concatenate to fit binarizer
        all_labels = list(itertools.chain(train_labels, test_labels))
        # Set overall length of dataset
        self.len: int = len(all_labels)
        if self.objective == "multilabel":
            # Fit the binarizer on all labels
            self.binarizer.fit(all_labels)
            # Binarize labels
            self.train_labels: np.ndarray = self.binarizer.transform(train_labels)
            self.test_labels: np.ndarray = self.binarizer.transform(test_labels)
        else:
            self.train_labels = train_labels
            self.test_labels = test_labels
        # Fetch text
        self.train_data: List[str] = [doc['text'] for doc in samples_train_list]
        self.test_data: List[str] = [doc['text'] for doc in samples_test_list]

    def __len__(self):
        return self.len

    def get_split(self):
        # --- Linux Bugs / Web of Science / Amazon ---
        if self.dataset_name == "bugs" or self.dataset_name == "wos" or self.dataset_name == "amazon":
            for train_index, test_index in self.splitter.split(self.data, self.labels_for_splitter):
                self.train_data, self.test_data = (itemgetter(*train_index)(self.data),
                                                   itemgetter(*test_index)(self.data))
                self.train_labels, self.test_labels = (itemgetter(*train_index)(self.labels),
                                                       itemgetter(*test_index)(self.labels))
                train = list(self.train_data), np.array(self.train_labels)
                test = list(self.test_data), np.array(self.test_labels)
                yield train, test
        # --- Blurb Genre Collection / RCV1-v2 ---
        elif self.dataset_name == "bgc" or self.dataset_name == "rcv1":
            train = self.train_data, self.train_labels
            test = self.test_data, self.test_labels
            yield train, test

    def get_split_with_indices(self):
        # --- Linux Bugs / Web of Science / Amazon ---
        if self.dataset_name == "bugs" or self.dataset_name == "wos" or self.dataset_name == "amazon":
            for train_index, test_index in self.splitter.split(self.data, self.labels_for_splitter):
                self.train_data, self.test_data = (itemgetter(*train_index)(self.data),
                                                   itemgetter(*test_index)(self.data))
                self.train_labels, self.test_labels = (itemgetter(*train_index)(self.labels),
                                                       itemgetter(*test_index)(self.labels))
                train = list(self.train_data), np.array(self.train_labels)
                test = list(self.test_data), np.array(self.test_labels)
                yield train, test, train_index
        # --- Blurb Genre Collection / RCV1-v2 ---
        elif self.dataset_name == "bgc" or self.dataset_name == "rcv1":
            train = self.train_data, self.train_labels
            test = self.test_data, self.test_labels
            yield train, test, None


def main():
    ds = DatasetManager("wos", {"stratifiedCV": 3,
                                "n_repeats": 1,
                                "objective": "multiclass"})
    i = 1
    for train, test in ds.get_split():
        print(i)
        i += 1
    ds = DatasetManager("rcv1", {"stratifiedCV": 3,
                                 "n_repeats": 1,
                                 "objective": "multiclass"})
    i = 1
    for train, test in ds.get_split():
        print(i)
        i += 1
    ds = DatasetManager("bgc", {"stratifiedCV": 3,
                                "n_repeats": 1,
                                "objective": "multiclass"})
    i = 1
    for train, test in ds.get_split():
        print(i)
        i += 1


if __name__ == "__main__":
    main()
