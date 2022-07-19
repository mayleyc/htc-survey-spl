from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from src.dataset_tools.prepare_financial_dataset import read_dataset as read_financial
from src.dataset_tools.prepare_linux_dataset import read_dataset as read_bugs
from src.model_evaluation.metrics import compute_metrics
from src.models.tfidf.preprocessing import process_flattened_dataset
from src.utils.torch_train_eval.generic_functions import dump_yaml, load_yaml


def run_svm_classifier(train_x, train_y, test_x, config):
    # Linear SVC is generally faster than SVM(kernel=linear)
    clf_svc = OneVsRestClassifier(LinearSVC(), n_jobs=6)
    # Create grid search setup and fit it
    grid_clf = GridSearchCV(clf_svc, config['SVM_GRID_PARAMS'], cv=config['gridsearchCV_SPLITS'],
                            scoring='f1_macro', verbose=True)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_params_)
    if config["retrain"]:
        # Retrain on "whole" data (grid_search splits similarly to above), utilizing the best parameters only
        optimized_clf = OneVsRestClassifier(LinearSVC(C=grid_clf.best_params_["estimator__C"],
                                                      max_iter=grid_clf.best_params_["estimator__max_iter"]),
                                            n_jobs=6)
        optimized_clf.fit(train_x, train_y)
        y_pred = optimized_clf.predict(test_x)
    else:
        # Same as just using best parameters w/o retraining
        y_pred = grid_clf.predict(test_x)
    return y_pred


def run_nb_classifier(train_x, train_y, test_x, config):
    clf_nb = MultinomialNB()
    # Create grid search setup and fit it

    grid_clf = GridSearchCV(clf_nb, config['NB_GRID_PARAMS'], cv=config['gridsearchCV_SPLITS'],
                            scoring='f1_macro', verbose=True)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_params_)
    if config["retrain"]:
        # Retrain on "whole" data (grid_search splits similarly to above), utilizing the best parameters only
        optimized_clf = OneVsRestClassifier(MultinomialNB(alpha=grid_clf.best_params_["alpha"],),
                                            n_jobs=6)
        optimized_clf.fit(train_x, train_y)
        y_pred = optimized_clf.predict(test_x)
    else:
        # Same as just using best parameters w/o retraining
        y_pred = grid_clf.predict(test_x)
    return y_pred


def run_rf_classifier(train_x, train_y, test_x, config):
    # clf_svc = OneVsRestClassifier(LinearSVC(), n_jobs=6)
    clf_rf = RandomForestClassifier(n_jobs=6)
    # Create grid search setup and fit it
    grid_clf = GridSearchCV(clf_rf, config['RANDOM_FOREST_GRID_PARAMS'],
                            cv=config['gridsearchCV_SPLITS'],
                            scoring='f1_macro', verbose=True)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_params_)
    if config["retrain"]:
        # Retrain on "whole" data (grid_search splits similarly to above), utilizing the best parameters only
        optimized_clf = RandomForestClassifier(
            bootstrap=grid_clf.best_params_["bootstrap"],
            # max_depth=grid_clf.best_params_["max_depth"],
            max_features=grid_clf.best_params_["max_features"],
            min_samples_leaf=grid_clf.best_params_["min_samples_leaf"],
            min_samples_split=grid_clf.best_params_["min_samples_split"],
            n_estimators=grid_clf.best_params_["n_estimators"],
            n_jobs=6
        )
        optimized_clf.fit(train_x, train_y)
        y_pred = optimized_clf.predict(test_x)
    else:
        # Same as just using best parameters w/o retraining
        y_pred = grid_clf.predict(test_x)
    return y_pred


def run_ab_classifier(train_x, train_y, test_x, config):
    # clf_svc = OneVsRestClassifier(LinearSVC(), n_jobs=6)
    # NOTE: I'm maxing at 85 as it's the highest number of labels
    clf_ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=85),
                                n_estimators=300, learning_rate=1)
    # Create grid search setup and fit it
    grid_clf = GridSearchCV(clf_ab, config['ADABOOST_GRID_PARAMS'],
                            cv=config['gridsearchCV_SPLITS'],
                            scoring='f1_macro', verbose=True)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_params_)
    if config["retrain"]:
        # Retrain on "whole" data (grid_search splits similarly to above), utilizing the best parameters only
        optimized_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, learning_rate=1)
        optimized_clf.fit(train_x, train_y)
        y_pred = optimized_clf.predict(test_x)
    else:
        # Same as just using best parameters w/o retraining
        y_pred = grid_clf.predict(test_x)
    return y_pred


def classify_flattened(tickets: List[List[str]], labels: List[str], config: Dict):
    # Variables that handle folds
    fold_i = 0
    fold_tot: int = config["stratifiedCV"]

    # Data and labels
    tickets: np.ndarray = np.array(tickets, dtype=object)
    labels: np.ndarray = np.array(labels, dtype=object)

    # Start K-Fold CV
    results: List = list()
    n_repeats: int = 2
    splitter: RepeatedStratifiedKFold = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=n_repeats,
                                                                random_state=772361728)

    # Operate on each split
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        print(f"Fold {fold_i}/{fold_tot * n_repeats} ({fold_tot} folds * {n_repeats} repeats)")
        # Assing split indices
        x_train, x_test = tickets[train_index], tickets[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # Create feature vectors
        # It's already tokenized; we tell it to return identity

        vectorizer = TfidfVectorizer(max_features=config["MAX_FEATURES"], ngram_range=(1, 2))
        # Train the feature vectors
        train_vectors = vectorizer.fit_transform([" ".join(x) for x in x_train])
        test_vectors = vectorizer.transform([" ".join(x) for x in x_test])

        y_pred = run_svm_classifier(train_vectors, y_train, test_vectors, config)
        # y_pred = run_rf_classifier(train_vectors, y_train, test_vectors, config)
        # y_pred = run_ab_classifier(train_vectors, y_train, test_vectors, config)
        # y_pred = run_nb_classifier(train_vectors, y_train, test_vectors, config)
        # Compute metrics with sklearn

        metrics = compute_metrics(y_test, y_pred, False)

        # Save metric for current fold
        results.append(metrics)

    # Average metrics over all folds and save them to csv
    df = pd.DataFrame(results)
    df.loc["avg", :] = df.mean(axis=0)
    results_folder: Path = (CONFIG.parent / "results")
    results_folder.mkdir(exist_ok=True)
    df.to_csv(results_folder / f"test_results.csv")
    dump_yaml(config, results_folder / f"test_config.yml")


def prepare_train_test_data(dataset: str) -> Tuple[List[List[str]], List[str]]:
    if dataset == "financial":
        df = read_financial()
    else:
        df = read_bugs()

    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    data, flattened_labels = process_flattened_dataset(df, remove_garbage=True, stop_words_removal=True)
    print("Preprocessing complete.\n")
    return data, flattened_labels


dataset_name: str = "bugs"
CONFIG = Path("src") / "models" / "tfidf" / "configs"


def main():
    config: Dict = load_yaml(CONFIG / "traditional_config.yml")
    data, labels = prepare_train_test_data(dataset=dataset_name)
    classify_flattened(data, labels, config)


if __name__ == "__main__":
    main()
