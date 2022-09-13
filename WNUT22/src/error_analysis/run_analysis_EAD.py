# Imports for SHAP MimicExplainer with LightGBM surrogate model
import pandas as pd
from raiwidgets import ErrorAnalysisDashboard
# Split data into train and test
from sklearn.model_selection import RepeatedStratifiedKFold


from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights


from src.dataset_tools.data_preparation.prepare_linux_dataset import read_dataset as read_bugs
from src.models.multi_level.modules.utility_functions import _predict, _setup_training
from src.models.multi_level.multilevel_models import MultiLevelBERT, SupportedBERT
from src.training_scripts.multilevel_models.train_multilevel import ensemble_args_split, get_actual_dump_name
from src.utils.generic_functions import load_yaml
import re

def example():
    from sklearn.datasets import load_wine
    from sklearn import svm

    # Imports for SHAP MimicExplainer with LightGBM surrogate model
    # from interpret.ext.blackbox import MimicExplainer
    # from interpret.ext.glassbox import LGBMExplainableModel

    wine = load_wine()
    X = wine['data']
    y = wine['target']
    classes = wine['target_names']
    feature_names = wine['feature_names']

    # Split data into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    from sklearn.linear_model import LogisticRegression
    clf = svm.SVC(gamma=0.001, C=100., probability=True)
    model = clf.fit(X_train, y_train)

    # Notice the model makes a fair number of errors
    print("number of errors on test dataset: " + str(sum(model.predict(X_test) != y_test)))

    from raiwidgets import ErrorAnalysisDashboard
    predictions = model.predict(X_test)
    # features (89, 13) binary (89,) (num_feats 13) binary (89,)
    ErrorAnalysisDashboard(dataset=X_test, true_y=y_test, features=feature_names, pred_y=predictions)

    # from interpret_community.common.constants import ModelTask
    # # Train the LightGBM surrogate model using MimicExplaner
    # model_task = ModelTask.Classification
    # explainer = MimicExplainer(model, X_train, LGBMExplainableModel,
    #                            augment_data=True, max_num_of_augmentations=10,
    #                            features=feature_names, classes=classes, model_task=model_task)
    #
    # # Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
    # # X_train can be passed as well, but with more examples explanations will take longer although they may be more accurate
    # global_explanation = explainer.explain_global(X_test)
    #
    # # Print out a dictionary that holds the sorted feature importance names and values
    # print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))
    #
    # ErrorAnalysisDashboard(global_explanation, model, dataset=X_test, true_y=y_test)


def main():
    df = read_bugs()
    config_path = "configs/error_analysis/support_config_error.yml"
    seeds = load_yaml("configs/random_seeds.yml")
    config = load_yaml(config_path)

    fold_tot = config["NUM_FOLD"]
    repeats = config["CV_REPEAT"]

    tickets = df["message"]
    labels_all = labels = df[config["LABEL"]]
    if "ALL_LABELS" in config:
        labels_all: pd.DataFrame = df[config["ALL_LABELS"]]

    EPOCH = 2  # HARDCODED
    fold_i = 0

    splitter = RepeatedStratifiedKFold(n_splits=fold_tot, n_repeats=repeats,
                                       random_state=seeds["stratified_fold_seed"])
    for train_index, test_index in splitter.split(tickets, labels):
        fold_i += 1
        x_train, x_test = tickets.iloc[train_index], tickets.iloc[test_index]
        y_train, y_test = labels_all.iloc[train_index], labels_all.iloc[test_index]

        x_train = x_train[:2000]
        x_test = x_test[:2000]
        y_train = y_train[:2000]
        y_test = y_test[:2000]

        # --------------------------------

        if "MODEL_L1" in config.keys():
            l1_dump = config['MODEL_L1']
            print(f"L1: {l1_dump}")
        if "MODEL_L2" in config.keys():
            l2_dump = config['MODEL_L2']
            print(f"L2: {l2_dump}")
        else:
            l2_dump = None  # for supported

        split_fun = lambda x: ensemble_args_split(l1_dump, l2_dump, EPOCH, x)
        config.update(split_fun(fold_i))
        config["PATH_TO_RELOAD"] = get_actual_dump_name(config["PATH_TO_RELOAD"], re.compile(f"fold_{fold_i}.*"))
        if config["mode"] == "Multilevel_ML":
            model_class = MultiLevelBERT
        elif config["mode"] == "Supported":
            model_class = SupportedBERT
        else:
            raise ValueError

        trainer, _, test_loader = _setup_training(train_config=config, model_class=model_class,
                                                  workers=0,
                                                  data=x_train, labels=y_train,
                                                  data_val=x_test, labels_val=y_test)

        y_pred, y_true = _predict(trainer.model, test_loader)  # (samples, num_classes)

        # assert (y_true == y_test).all()
        # FIXME: check
        # KINDA RIGHT BUT KINDA WRONG CAUSE OF SPLIT
        # feature_names = list(y_train.unique())
        y_pred = y_pred.argmax(axis=-1)
        rai_insights = RAIInsights(model, train_data, test_data, target_feature, 'regression',
                                   categorical_features=[])
        # Interpretability
        rai_insights.explainer.add()
        # Error Analysis
        rai_insights.error_analysis.add()
        # Counterfactuals: accepts total number of counterfactuals to generate, the range that their label should fall under,
        # and a list of strings of categorical feature names
        rai_insights.counterfactual.add(total_CFs=20, desired_range=[50, 120])
        rai_insights.compute()
        ResponsibleAIDashboard(rai_insights)

        # ErrorAnalysisDashboard(dataset=x_test, true_y=y_true, pred_y=y_pred)
        # ----------------------------
        input("Check it")

    # model_task = ModelTask.Classification
    # explainer = MimicExplainer(model, X_train, LGBMExplainableModel,
    #                            augment_data=True, max_num_of_augmentations=10,
    #                            features=feature_names, classes=classes, model_task=model_task)
    #
    # global_explanation = explainer.explain_global(X_test)
    #
    # # Print out a dictionary that holds the sorted feature importance names and values
    # print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))
    # # ----------------------------
    # ErrorAnalysisDashboard(global_explanation, model, dataset=X_test, true_y=y_test)
    # # ----------------------------
    input("Check it")

def main():
    pass

if __name__ == "__main__":
    main()
    # example()
