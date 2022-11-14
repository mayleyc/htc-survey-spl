import os
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import src.models.HiMatch.helper.logger as logger
from src.dataset_tools.dataset_manager import DatasetManager
from src.models.HiAGM import Vocab, Configure, _predict, \
    set_optimizer
from src.models.HiMatch.data_modules.data_loader import data_loaders
from src.models.HiMatch.data_modules.vocab import Vocab
from src.models.HiMatch.helper.adamw import AdamW
from src.models.HiMatch.helper.configure import Configure
from src.models.HiMatch.helper.lr_schedulers import get_linear_schedule_with_warmup
from src.models.HiMatch.helper.utils import load_checkpoint
from src.models.HiMatch.models.model import HiMatch
from src.models.HiMatch.prepare_our_data import write_jsonl_split
from src.models.HiMatch.train_modules.criterions import ClassificationLoss, MarginRankingLoss
from src.models.HiMatch.train_modules.trainer import Trainer
from src.training_scripts.script_utils import save_results
from src.utils.metrics import compute_metrics, compute_hierarchical_metrics

os.linesep = '\n'


def run_training(config: Configure, dataset: str, out_folder: Path):
    ds_manager = DatasetManager(dataset_name=dataset, training_config=config.dict)
    os.makedirs(out_folder, exist_ok=True)
    containing_folder = out_folder
    results = list()
    # Train in splits
    fold_i: int = 0
    for (x_train, y_train), (x_test, y_test) in ds_manager.get_split():
        fold_i += 1
        out_folder = containing_folder / f"fold {fold_i}"
        config['train'].checkpoint.dir = out_folder / "checkpoints"
        os.makedirs(out_folder, exist_ok=True)
        # Get validation data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=ds_manager.seeds['validation_split_seed'],
                                                          stratify=y_train)
        # Loading corpus and generate vocabulary
        corpus_vocab: Vocab = write_jsonl_split(config, (x_train, ds_manager.binarizer.inverse_transform(y_train)),
                                                (x_test, ds_manager.binarizer.inverse_transform(y_test)),
                                                (x_val, ds_manager.binarizer.inverse_transform(y_val)))
        tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_model"], do_lower_case=True)
        # Get data
        train_loader, val_loader, test_loader, label_desc_loader = data_loaders(config, corpus_vocab,
                                                                                bert_tokenizer=tokenizer)
        print(f"Building model for fold {fold_i}.")
        # build up model
        himatch = HiMatch(config, corpus_vocab, model_mode='TRAIN')
        himatch.to(config['train'].device_setting.device)
        # define training objective & optimizer
        criterion = ClassificationLoss(os.path.join(config['data'].data_dir, config['data'].hierarchy),
                                       corpus_vocab.v2i['label'],
                                       recursive_penalty=config['train'].loss.recursive_regularization.penalty,
                                       recursive_constraint=config['train'].loss.recursive_regularization.flag,
                                       loss_type="bce")
        # define ranking loss
        criterion_ranking = MarginRankingLoss(config)

        if "bert" in config["model"].type:
            t_total = int(len(train_loader) * (config["train"].end_epoch - config["train"].start_epoch))

            param_optimizer = list(himatch.named_parameters())
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            warmup_steps = int(t_total * 0.1)
            optimizer = AdamW(optimizer_grouped_parameters, lr=config["train"].optimizer.learning_rate, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
        else:
            optimizer = set_optimizer(config, himatch)
            scheduler = None

        # get epoch trainer
        trainer = Trainer(model=himatch,
                          criterion=[criterion, criterion_ranking],
                          optimizer=optimizer,
                          vocab=corpus_vocab,
                          tokenizer=tokenizer,
                          scheduler=scheduler,
                          config=config,
                          label_desc_loader=label_desc_loader,
                          train_loader=train_loader,
                          dev_loader=val_loader,
                          test_loader=test_loader)

        # set origin log
        model_checkpoint = Path(config.train.checkpoint.dir)
        if not os.path.isdir(model_checkpoint):
            model_checkpoint.mkdir(parents=True)
            # os.mkdir(model_checkpoint)
            config.train.start_epoch = 0
        else:
            # loading previous checkpoint
            dir_list = os.listdir(model_checkpoint)
            dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))
            latest_model_file = 'best_micro_HiMatch'
            if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
                logger.info('Loading Previous Checkpoint...')
                logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
                best_performance_dev, config = load_checkpoint(
                    model_file=os.path.join(model_checkpoint, latest_model_file),
                    model=himatch,
                    config=config,
                    optimizer=optimizer)
                logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                    best_performance_dev[0], best_performance_dev[1]))

        # train
        trainer.run_train()

        # Use the model to predict test/validation samples
        y_pred, y_true = _predict(trainer.model, val_loader)  # (samples, num_classes)

        # Compute metrics with sklearn
        metrics = compute_metrics(y_true, y_pred, argmax_flag=False)
        # hiagm.label_map

        h_metrics = compute_hierarchical_metrics(y_true, y_pred, encoder_dump_or_mapping=himatch.vocab,
                                                 taxonomy_path=Path(config['data']['data_dir'])
                                                 / config['data']['hierarchy'])
        # Save metric for current fold
        # requires python 3.9
        all_metrics = metrics | h_metrics  # join
        results.append(all_metrics)

    save_results(results, containing_folder, config.dict)


def run_configuration():
    model_name: str = "HiMatch"
    # Paths
    config_base_path: Path = Path("config") / model_name
    output_path: Path = Path("dumps") / model_name
    config_list: List = [# "himatch_config_blurb.json",
                         "himatch_config_wos.json",
                         "himatch_config_bugs.json",
                         "himatch_config_rcv1.json",
                         "himatch_config_amazon.json", ]

    for c in config_list:
        # Funny, depressing, yet necessary.
        while True:
            try:
                # Prepare configuration
                config_path: Path = (config_base_path / c)
                config = Configure(config_json_file=config_path)
                logger.Logger(config)

                if config["train"].device_setting.device == 'cuda':
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["train"].device_setting.visible_device_list)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ''

                print(f"Specific model: {model_name}")
                print(f"Dataset: {config['data']['dataset']}")
                # Prepare output
                out_folder = output_path / config['data']['dataset']
                # Train
                run_training(config=config,
                             dataset=config["data"]["dataset"],
                             out_folder=out_folder)
            except RuntimeError:
                print("Zero loss. Trying again...")
            else:
                break


if __name__ == "__main__":
    run_configuration()
