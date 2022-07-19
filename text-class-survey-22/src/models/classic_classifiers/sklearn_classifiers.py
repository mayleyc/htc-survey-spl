import logging
from ast import literal_eval
from typing import Type, Dict

import joblib
import numpy as np
import regex as re
import sklearn.metrics as skmetrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import src.models.classic_classifiers.text_preprocessing as txt_preproc
from src.datasets.generation.base import BaseDataset


class ClassicHandler:
    def __init__(self, dataset: Type[BaseDataset], run_config: Dict):
        self.dataset = dataset
        self.train_path: str = str(dataset.generated_root(run_config['DATASET_NAME']) / "train.txt")
        self.test_path: str = str(dataset.generated_root(run_config['DATASET_NAME']) / "test.txt")
        encoder_path: str = str(dataset.generated_root(run_config['DATASET_NAME']) / 'encoder.bin.xz')
        self.__decoder = joblib.load(encoder_path)
        self.config = run_config
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def __load_process_save(self, path, name):
        x, y = self.__read_data(path)
        # Preprocessing test into tokenized versions
        x = txt_preproc.pipeline(x, self.dataset.LANG, logging=True)
        with open(self.dataset.generated_root(self.config['DATASET_NAME']) / name, "w+", encoding='utf-8') as f:
            lines = [f"{line}\n" for line in x]
            f.writelines(lines)
        return x, y

    def __load_lemmatized(self, path, name):
        print("[INFO] Found lemmatized set saved")
        _, y = self.__read_data(path)
        x = [literal_eval(line) for line in open(self.dataset.generated_root(self.config['DATASET_NAME']) / name, 'r',
                                                 encoding='utf-8').readlines()]
        return x, y

    def prepare_data(self, load_lemmatized: bool = True, **kwargs):
        """
        Loads and prepares data into class attributes. Applies preprocessing pipeline to text, unless and already
        preprocessed file exists. Then, turns processed text into vectors with TFIDF vectorizer.
        """
        # *************** TRAIN ***************
        train_lemmatized: bool = (
                    self.dataset.generated_root(self.config['DATASET_NAME']) / "train_lemmatized.txt").exists()
        test_lemmatized: bool = (
                    self.dataset.generated_root(self.config['DATASET_NAME']) / "test_lemmatized.txt").exists()
        # Check for a lemmatized, saved version and load it if its there. Otherwise, create it
        if load_lemmatized:
            if train_lemmatized:
                self.train_x, self.train_y = self.__load_lemmatized(self.train_path, "train_lemmatized.txt")
            else:
                self.train_x, self.train_y = self.__load_process_save(self.train_path, "train_lemmatized.txt")
            if test_lemmatized:
                self.test_x, self.test_y = self.__load_lemmatized(self.test_path, "test_lemmatized.txt")
            else:
                self.test_x, self.test_y = self.__load_process_save(self.test_path, "test_lemmatized.txt")
        else:
            # Run preprocessing (without loading or saving) with kwargs arguments
            self.train_x, self.train_y = self.__read_data(self.train_path)
            self.train_x = txt_preproc.pipeline(self.train_x, self.dataset.LANG, **kwargs)
            self.test_x, self.test_y = self.__read_data(self.test_path)
            self.test_x = txt_preproc.pipeline(self.test_x, self.dataset.LANG, **kwargs)
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=self.config['MAX_FEATURES'], ngram_range=(1, 2))
        train_joint_tokens = [" ".join(x) for x in self.train_x]
        test_joint_tokens = [" ".join(x) for x in self.test_x]
        vectorizer.fit(train_joint_tokens)
        self.train_x = vectorizer.transform(train_joint_tokens)
        self.test_x = vectorizer.transform(test_joint_tokens)

    def apply_classifier(self, clf, parameters: Dict = None, name: str = 'default'):
        """
        Applies a shallow classifier to prepared data.

        Parameters
        ----------
        clf : a classifier suitable for the data
        parameters : Grid search parameters. If None, no search is performed
        name : name of the classifier (for display purposes)
        """
        if parameters:
            clf = GridSearchCV(clf, parameters, cv=self.config['CV'], scoring='f1_macro', verbose=True)
        clf.fit(self.train_x, self.train_y)
        if parameters:
            print(clf.best_params_)
        prediction = clf.predict(self.test_x)

        logging.info(f"*** {name.title()} results ***")
        logging.info(f"Accuracy: {skmetrics.accuracy_score(self.test_y, prediction):.4f}")
        logging.info(f"Precision: {skmetrics.precision_score(self.test_y, prediction, average='macro'):.4f}")
        logging.info(f"Recall: {skmetrics.recall_score(self.test_y, prediction, average='macro'):.4f}")
        logging.info(f"F1: {skmetrics.f1_score(self.test_y, prediction, average='macro'):.4f}")
        logging.info("\n")

    def __read_data(self, path: str):
        """
        Read FastText formatted txt files

        Parameters
        ----------
        path : path to file

        Returns
        -------
        Text (as str), labels (as ndarray of binarized labels)
        """
        x = []
        y = []
        for line in open(path, "r", encoding='utf-8').readlines():
            tagged_labels = re.findall(r'__label__\w+', line)
            text = line[len(" ".join(tagged_labels)) + 1:]
            labels = self.__decoder.transform([[cat[len('__label__'):] for cat in tagged_labels]])[0]
            x.append(text)
            y.append(labels)
        return x, np.asarray(y)

# def main():
#     dataset = EnWiki
#     print(f"Current language: {dataset.LANG.title()}")
#     # --- CLASSIFY ---
#     # Linear kernel SVM
#     clf_svc = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
#     # Naive Bayes
#     clf_nb = OneVsRestClassifier(MultinomialNB(), n_jobs=-1)
#     # multi-label adapted kNN classifier with bayesian prior corrections
#     # http://scikit.ml/api/skmultilearn.adapt.mlknn.html#skmultilearn.adapt.MLkNN
#     # clf_mlknn = MLkNN(k=3)
#     # Prepare classification
#     handler = ClassicHandler(dataset)
#     handler.prepare_data(load_lemmatized=False, disable=['bigrams', 'trigrams', 'lemma', 'stem'], logging=True)
#
#     parameters = {'estimator__C': range(1, 5), 'estimator__max_iter': [1000]}
#     start = time.perf_counter()
#     handler.apply_classifier(clf_svc, parameters, 'Linear SVC')
#     end = time.perf_counter()
#     print(f"Linear SVC: {end - start:0.2f}")
#
#     parameters = {'estimator__alpha': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 10.0, ],
#                    'estimator__fit_prior': [True, False]}
#     handler.apply_classifier(clf_nb, parameters, 'Multinomial Naive Bayes')
#     # Skip future warnings
#     # with warnings.catch_warnings():
#     #     warnings.simplefilter(action='ignore', category=FutureWarning)
#     #     # parameters = {'k': range(2, 5), 's': [0.5, 0.7, 1.0]}
#     #     handler.apply_classifier(clf_mlknn, None, 'MLKNN')
#
#
# if __name__ == "__main__":
#     main()
