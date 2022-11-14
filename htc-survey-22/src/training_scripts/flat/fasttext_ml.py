import datetime as dt
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List

from fasttext import train_supervised
from sklearn.model_selection import train_test_split

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.FastText_flat.fasttext_flat import save_to_fasttext_format, get_model_parameters, _predict
from src.models.SVM.preprocessing import process_list_dataset
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml


def run_training(config: Dict, dataset: str, out_folder: Path):
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config)
    os.makedirs(out_folder, exist_ok=True)
    results = list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in ds_manager.get_split():
        fold_i += 1
        print("Starting preprocessing ...")

        # x_train = x_train[:1000]
        # y_train = y_train[:1000]
        # x_test = x_test[:1000]
        # y_test = y_test[:1000]

        x_train, x_test = process_list_dataset(x_train, x_test,
                                               remove_garbage=config["remove_garbage"],
                                               stop_words_removal=config["stop_words_removal"])

        print("Preprocessing complete.\n")
        print(f"Building model for fold {fold_i}.")
        emb_args = dict()
        pre_trained_vecs = config.get("PRE_TRAINED_VECTORS", None)
        if pre_trained_vecs is not None:
            emb_args = dict(pretrainedVectors=pre_trained_vecs)
        # Create folder for models / splits
        out_folder.mkdir(exist_ok=True, parents=True)
        # ---------------------------------------
        # Divide in folders to keep data
        fold_folder: Path = out_folder / f"fold_{fold_i}"
        fold_folder.mkdir(exist_ok=True)
        # ---------------------------------------
        train_data_path: Path = fold_folder / "train.txt"  # Training data
        train_data_val_path: Path = fold_folder / "train_val.txt"  # Training set without validation data
        val_data_path: Path = fold_folder / "val.txt"  # Validation data
        test_data_path: Path = fold_folder / "test.txt"  # Test set data
        model_data_path: Path = fold_folder / "model.bin"
        output_parameters_path: Path = fold_folder / "params.json"
        # Dump the current fold in FT format
        save_to_fasttext_format(x_train, ds_manager.binarizer.inverse_transform(y_train), train_data_path)
        save_to_fasttext_format(x_test, ds_manager.binarizer.inverse_transform(y_test), test_data_path)
        # ---------------------------------------
        if config["autoTune"]:
            # Notice that the test set will be ignored, and never used in validation or training
            # Test results will be done after
            # print(f"# Samples: {len(x_train)}")
            # if dataset in ["rcv1", "wos"]:
            #     # Fix too few classes in val set by oversampling
            #     k_to_fix = 2
            #     dup_n = 2
            #     low_freq_classes_idx = np.argpartition(y_train.sum(0), k_to_fix)[:k_to_fix]
            #     samples_to_duplicate_indexes = (y_train[:, low_freq_classes_idx] > 0).nonzero()[0]
            #     samples_to_duplicate_indexes[::-1].sort()
            #     for idx in samples_to_duplicate_indexes:
            #         # [x_train.insert(idx + 1, x_train[idx])]
            #         x_train = np.insert(x_train, [idx + 1] * dup_n, x_train[idx], axis=0).tolist()
            #         y_train = np.insert(y_train, [idx + 1] * dup_n, y_train[idx], axis=0)
            #         lab = ds_manager.labels_for_splitter[idx]
            #         [ds_manager.labels_for_splitter.insert(idx + 1, lab) for _ in range(dup_n)]
            # print(f"Min samples per cat: {y_train.sum(0).min()}")
            # print(f"# Samples after: {len(x_train)}")
            fs: int = 0 if dataset in {"rcv1"} else -1
            stratify_labels = [a[fs] for a in ds_manager.binarizer.inverse_transform(y_train)]
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                              random_state=ds_manager.seeds['validation_split_seed'],
                                                              stratify=stratify_labels,
                                                              shuffle=True)
            # Overwrites train data to remove validation data
            save_to_fasttext_format(x_train, ds_manager.binarizer.inverse_transform(y_train), train_data_val_path)
            save_to_fasttext_format(x_val, ds_manager.binarizer.inverse_transform(y_test), val_data_path)

            model = train_supervised(
                input=str(train_data_val_path),
                autotuneValidationFile=str(val_data_path),
                autotuneDuration=config["autoTuneDuration"],
                loss="ova",
                **emb_args
            )

            # Save best hyperparameters
            best_hyperparams = get_model_parameters(model, config["ft_params"])
            pprint(best_hyperparams)
            with open(output_parameters_path, "w") as f:
                json.dump(best_hyperparams, f)

            model = train_supervised(input=str(train_data_path), **best_hyperparams)
        # ---------------------------------------
        else:
            model = train_supervised(input=str(train_data_path), **emb_args)
        # ---------------------------------------
        model.save_model(str(model_data_path))
        # Find metrics on TEST set for the resulting model OF THIS FOLD
        metrics = _predict(model, str(test_data_path), config["taxonomy_path"], enc=ds_manager.binarizer)
        del model
        results.append(metrics)
        # ---------------------------------------
        save_results(results, out_folder, config)


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "FastText"
    output_path: Path = Path("dumps") / "FastText"
    config_list: List = ["ft_config_bgc.yml", "ft_config_bugs.yml", "ft_config_rcv1.yml", "ft_config_wos.yml", "ft_config_amz.yml"]

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        # config["n_repeats"] = 1
        specific_model = f"ML_FastText"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # Train
        run_training(config=config,
                     dataset=config["dataset"],
                     out_folder=out_folder)


if __name__ == "__main__":
    run_configuration()
