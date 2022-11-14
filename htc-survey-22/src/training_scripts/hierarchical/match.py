import os
from logging import getLogger
from operator import itemgetter
from pathlib import Path
from typing import List, Dict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.Match.deepxml.additional_utils import get_data_dump_new
from src.models.Match.deepxml.dataset import MultiLabelDataset
from src.models.Match.deepxml.match import MATCH
from src.models.Match.deepxml.models import Model
from src.training_scripts.script_utils import save_results
from src.utils.generic_functions import load_yaml
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics

os.linesep = '\n'


def run_training(config: Dict, dataset: str, out_folder: Path):
    model_cnf, data_cnf = config["model"], config["data"]
    reg: bool = False
    edges = set()
    # ---------------
    logger = getLogger()
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config)
    os.makedirs(out_folder, exist_ok=True)
    results = list()
    # Train in splits
    fold_i: int = 0
    device = torch.device(model_cnf["device"])
    containing_folder = out_folder

    for (x_train, y_train), (x_test, y_test), train_index in ds_manager.get_split_with_indices():
        fold_i += 1
        model_folder_ = containing_folder / f"fold_{fold_i}"

        # FOR DEBUG
        # x_train = x_train[:100]
        # x_test = x_test[:100]
        # y_train = y_train[:100]
        # y_test = y_test[:100]

        # y_train = ds_manager.ml_binarizer.inverse_transform(y_train)
        # y_test = ds_manager.ml_binarizer.inverse_transform(y_test)

        x_train, kv_train = get_data_dump_new(model_cnf, x_train, base_dump=Path(data_cnf["data_dump_train"]),
                                              fold=fold_i)
        x_test, kv_test = get_data_dump_new(model_cnf, x_test, base_dump=Path(data_cnf["data_dump_test"]), fold=fold_i)
        emb_init = kv_train.syn1neg

        labels_num = len(ds_manager.binarizer.classes_)

        # Validation split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=ds_manager.seeds["validation_split_seed"],
                                                          stratify=itemgetter(*train_index)(
                                                              ds_manager.labels_for_splitter))

        logger.info('Training')
        # -----------------------------------------
        train_ds = MultiLabelDataset(x_train, y_train, vectors=kv_train.wv, config=model_cnf)
        val_ds = MultiLabelDataset(x_val, y_val, training=True, vectors=kv_train.wv, config=model_cnf)
        test_ds = MultiLabelDataset(x_test, y_test, training=True, vectors=kv_train.wv, config=model_cnf)
        # -----------------------------------------
        train_loader = DataLoader(train_ds, model_cnf['train']['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, model_cnf['valid']['batch_size'], num_workers=0)
        test_loader = DataLoader(test_ds, model_cnf['valid']['batch_size'], num_workers=0)
        if reg:
            classes = ds_manager.ml_binarizer.classes_.tolist()
            with open(data_cnf['hierarchy']) as fin:
                for line in fin:
                    data = line.strip().split()
                    p = data[0]
                    if p not in classes:
                        continue
                    p_id = classes.index(p)
                    for c in data[1:]:
                        if c not in classes:
                            continue
                        c_id = classes.index(c)
                        edges.add((p_id, c_id))
            logger.info(F'Number of Edges: {len(edges)}')

        model = Model(network=MATCH, labels_num=labels_num, model_path=model_folder_, emb_init=emb_init, mode='train',
                      reg=reg, hierarchy=edges,
                      **data_cnf['model'], **model_cnf['model'],
                      device=device, label_binarizer=ds_manager.binarizer)
        model.train(train_loader, val_loader, **model_cnf['train'])
        logger.info('Finish Training')

        # Use the model to predict test/validation samples
        y_pred, y_true = model.predict(test_loader)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
        # hiagm.label_map

        h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=ds_manager.binarizer,
                                                 taxonomy_path=Path(config['data']["taxonomy_path"]))
        # Save metric for current fold
        # requires python 3.9
        all_metrics = metrics | h_metrics  # join
        results.append(all_metrics)

        save_results(results, containing_folder, config)


def run_configuration():
    # Paths
    config_base_path: Path = Path("config") / "Match"
    output_path: Path = Path("dumps") / "Match"
    config_list: List = ["match_config_bugs.yaml",
                         "match_config_wos.yaml",
                         "match_config_blurb.yaml",
                         "match_config_rcv1.yaml",
                         "match_config_amazon.yaml"]

    for c in config_list:
        config_path: Path = (config_base_path / c)
        config: Dict = load_yaml(config_path)

        specific_model = f"Match"
        print(f"Specific model: {specific_model}")
        print(f"Dataset: {config['data']['name']}")
        # Prepare output
        out_folder = output_path / config['data']['name']
        # Train
        run_training(config=config,
                     dataset=config["data"]["name"],
                     out_folder=out_folder)


if __name__ == "__main__":
    run_configuration()
