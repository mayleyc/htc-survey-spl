import logging
import os
import time
from typing import Dict

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.datasets.generation.nc import RCV1en, RCV2it, RCV2fr
from src.datasets.generation.tl import EnWiki, ItWiki, FrWiki
from src.models.classic_classifiers.sklearn_classifiers import ClassicHandler
from utils.general import load_yaml

folders = ["rcv2it-4", "rcv1en-4", "rcv2fr-4",
           "itwiki-4", "enwiki-4", "frwiki-4"]
datasets = [RCV2it, RCV1en, RCV2fr,
            ItWiki, EnWiki, FrWiki]


def classify(dataset, run_config):
    logging.info(f"Current language: {dataset.LANG.title()}")
    logging.info(f"Current dataset: {run_config['DATASET_NAME']}")
    print(f"Current language: {dataset.LANG.title()}")
    print(f"Current dataset: {run_config['DATASET_NAME']}")
    # --- CLASSIFY ---
    # Linear kernel SVM
    clf_svc = OneVsRestClassifier(LinearSVC(), n_jobs=run_config['NUM_JOBS'])
    # Naive Bayes
    clf_nb = OneVsRestClassifier(MultinomialNB(), n_jobs=run_config['NUM_JOBS'])
    # Prepare classification
    handler = ClassicHandler(dataset, run_config)
    handler.prepare_data(load_lemmatized=False, disable=['bigrams', 'trigrams', 'lemma', 'stem'], logging=True)

    start = time.perf_counter()
    handler.apply_classifier(clf_svc, run_config['SVM_GRID'], 'Linear SVC')
    end = time.perf_counter()
    print(f"Linear SVC time: {end - start:0.2f}")

    start = time.perf_counter()
    handler.apply_classifier(clf_nb, run_config['NB_GRID'], 'Multinomial Naive Bayes')
    end = time.perf_counter()
    print(f"NB time: {end - start:0.2f}")


def main():
    logging.basicConfig(filename=os.path.join("experiments", "classic_results.log"), level=logging.INFO)

    dataset = FrWiki
    # logging.info(f"Current language: {dataset.LANG.title()}")
    run_config: Dict = load_yaml('configs/classic_run.yml')
    # logging.info(f"Current dataset: {run_config['DATASET_NAME']}")
    if run_config['DATASET_NAME'] == "all":
        for d, c in zip(datasets, folders):
            run_config['DATASET_NAME'] = c
            classify(d, run_config)
    else:
        classify(dataset, run_config)


if __name__ == "__main__":
    main()
