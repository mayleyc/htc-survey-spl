import datetime as dt
import os
from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as td
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from torchmetrics import F1Score
from tqdm import tqdm

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.Capsules.dataset import EmbeddingDataset, load_vectors
from src.models.Capsules.loss import CapsuleLoss
from src.models.Capsules.model import CapsNet
from src.models.SVM.preprocessing import process_list_dataset
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml, get_model_dump_path
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics
from src.utils.torch_train_eval.early_stopper import EarlyStopping
from src.utils.torch_train_eval.evaluation import MetricSet
from src.utils.torch_train_eval.trainer import Trainer


def predict(model, data: td.DataLoader):
    model.train(False)
    y_pred = list()
    y_true = list()
    with torch.no_grad():
        for i, pred_data in tqdm(enumerate(data), total=len(data)):
            y_pred_t, y_true_t, *_ = model(pred_data)

            y_len = y_pred_t.norm(2, dim=2)  # (bs, categories)
            # pred = torch.where(pred > t, 1, 0)

            y_pred.append(y_len.detach().cpu().numpy())
            y_true.append(y_true_t.cpu().numpy())
    return np.concatenate(y_pred), np.concatenate(y_true)


def run_training(config: Dict, train_set: str, out_folder: Path, split_fun=None):
    ds_manager = DatasetManager(dataset_name=train_set, training_config=config)
    os.makedirs(out_folder, exist_ok=True)
    results = list()
    # Train in splits
    fold_i: int = 0
    seeds = load_yaml("config/random_seeds.yml")

    # Create feature vectors
    vectors, vocab = load_vectors(Path("data") / "glove.6B" / "glove.6B.300d.txt")
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = BertNormalizer(strip_accents=True, lowercase=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=config["NUM_TOKENS"])

    # Trainer config
    model_folder = config["MODEL_FOLDER"]

    for (x_train, y_train), (x_test, y_test), idx_train in ds_manager.get_split_with_indices():
        if config["validation"] is True:
            # Replace test set with validation set. Notice that the test set will be ignored,
            # and never used in validation or training
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                                random_state=seeds["validation_split_seed"],
                                                                stratify=itemgetter(*idx_train)(ds_manager.labels_for_splitter))
        # x_train = x_train[:100]
        # y_train = y_train[:100]
        # x_test = x_test[:100]
        # y_test = y_test[:100]
        fold_i += 1

        if split_fun is not None:
            config.update(split_fun(fold_i))

        print("Starting preprocessing ...")
        x_train, x_test = process_list_dataset(x_train, x_test,
                                               remove_garbage=True,
                                               stop_words_removal=True)
        print("Preprocessing complete.\n")
        config["MODEL_FOLDER"] = str(model_folder / f"fold_{fold_i}")
        # Setup model and optimizer
        early_stopper = EarlyStopping(*config["EARLY_STOPPING"].values())
        num_class = len(ds_manager.binarizer.classes_)
        network = CapsNet(
            emb_num=config["NUM_TOKENS"],
            vectors=vectors,
            capsule_layers_kwargs=[{"num_units": num_class, "out_size": 16, "routing_iter": config.get("routing_iterations", None)}],
            regularize=config["REGULARIZE"]
        )
        opt = torch.optim.AdamW(network.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["L2_REG"])
        loss = CapsuleLoss(config)
        metrics = MetricSet({"f1": (
            F1Score(num_classes=num_class, threshold=config["m_plus"], average="macro"),
            lambda tup, **kwargs: (tup[0].norm(2, dim=2), tup[1])
        )})
        t = Trainer(network, config, loss, opt, early_stopper, metrics=metrics, add_start_time_folder=False)
        # This preprocessing does tokenization and text cleanup
        # Simple filtering is done in read functions

        train_set = EmbeddingDataset(x_train, y_train, tokenizer)
        train_loader = td.DataLoader(train_set, shuffle=True, batch_size=config["BATCH_SIZE"])
        test_set = EmbeddingDataset(x_test, y_test, tokenizer)
        test_loader = td.DataLoader(test_set, shuffle=False, batch_size=config["TEST_BATCH_SIZE"])

        t.train(train_loader, test_loader)

        # TEST the model
        # First reload last improving epoch
        t.load_previous(t.last_saved_checkpoint, model_only=True)
        model = t.model

        y_pred, y_true = predict(model, test_loader)
        metrics = compute_metrics(y_true, y_pred, False, threshold=config["m_plus"])
        h_metrics = compute_hierarchical_metrics(y_true, y_pred,
                                                 encoder_dump_or_mapping=ds_manager.binarizer,
                                                 taxonomy_path=Path(config["taxonomy_path"]),
                                                 threshold=config["m_plus"])
        all_metrics = metrics | h_metrics  # join
        # Save metric for current fold
        results.append(all_metrics)
        # CLEANUP
        del t, model
        torch.cuda.empty_cache()
        # ---------------------------------------
        # Save results at each fold (overwrite)
        save_results(results, out_folder, config)


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "CapsNet" / "validation"
    output_path: Path = Path("dumps") / "CapsNet"
    # config_list: List = ["capsnet_val1.yml", "capsnet_val2.yml", "capsnet_val3.yml", "capsnet_val4.yml", "capsnet_val5.yml", "capsnet_val6.yml", "capsnet_val7.yml"]
    config_list: List = ["capsnet_val_rl.yml", "capsnet_val_mar.yml", "capsnet_val_mar_ga.yml"]

    for c in config_list:
        # Prepare configuration
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)
        specific_model = f"ML_CapsNet"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['dataset']}")
        # Prepare output
        out_folder = output_path / specific_model / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        config["MODEL_FOLDER"] = out_folder
        kw = dict()
        if config["RELOAD"] is True:
            reload_path = Path(config["PATH_TO_RELOAD"])
            kw = dict(split_fun=lambda f: get_model_dump_path(reload_path, f, config.get("EPOCH_RELOAD", None)))
            out_folder = reload_path
        # Train
        run_training(config=config,
                     train_set=config["dataset"],
                     out_folder=out_folder, **kw)


if __name__ == "__main__":
    run_configuration()
