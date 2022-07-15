import abc
import logging
import os
from typing import Dict, Union, Type

import pandas as pd
import torch
import torch.utils.data as td
import torchmetrics as tm
import transformers
from tokenizers import normalizers as norm, models as mod, pre_tokenizers as pretok, Tokenizer
from torch import nn

from src.datasets.generation.base import BaseDataset
from src.datasets.torch_dataset import TransformerIterableDataset, TransformerMapDataset, WordEmbeddingMapDataset, \
    iter_coll_fn_embed
from utils.embeddings import load_vectors
from utils.general import load_yaml
from utils.training.evaluation import Evaluator
from utils.training.model import BaseModel
from utils.training.trainer import Trainer


class ScriptHandler(abc.ABC):
    def __init__(self, dataset_class: Type[BaseDataset],
                 run_config: Union[str, Dict] = os.path.join("configs", "transformers_run.yml"),
                 model_config: Union[str, Dict] = os.path.join("configs", "transformers_models.yml")):
        """
        Handles generic scripts for training/evaluating experiments. Loads configurations from appropriate YAML files.

        Parameters
        ----------
        run_config : dict or str, training/evaluation parameters (as a dict or path to yaml file)
        model_config : dict or str, model specifics (as a dict or path to yaml file)
        """
        # Load accepts paths to YAML files or a dict
        # --- LOAD CONFIG ---
        if isinstance(run_config, str):
            self.run_config: Dict = load_yaml(run_config)
        elif isinstance(run_config, dict):
            self.run_config = run_config
        else:
            raise ValueError("Invalid configuration: either pass a path to yaml or a dict configuration.")
        # --- MODEL CONFIG ---
        if isinstance(model_config, str):
            self.model_config: Dict = load_yaml(model_config)
        elif isinstance(run_config, dict):
            self.model_config = model_config
        else:
            raise ValueError("Invalid configuration: either pass a path to yaml or a dict configuration.")
        # --- Initialize parameters ---
        self.datasetClass: Type[BaseDataset] = dataset_class
        self.checkpoint_path: str = os.path.join(self.run_config["MODELS_PATH"], self.datasetClass.__name__.lower(),
                                                 self.run_config["RUN_NAME"], self.run_config["CHECKPOINT_NAME"])
        self.__warning_run_overwrite = False
        # Create path to dump if it does not exist, else raise error
        _path = os.path.dirname(self.checkpoint_path)
        try:
            os.makedirs(_path, exist_ok=False)
        except OSError:
            if len(os.listdir(_path)) > 0:
                self.__warning_run_overwrite = True

        self.train_loader, self.eval_loader, self.test_loader = None, None, None

        self.loss_f = None
        self.train_config: Dict = {}

        self.net_config = None
        self.network = None

        self.metrics_f = None

    @abc.abstractmethod
    def initialize_test_dataset(self):
        pass

    @abc.abstractmethod
    def initialize_train_dataset(self):
        pass

    @abc.abstractmethod
    def init_model(self):
        pass

    def initialize_configuration(self) -> None:
        """
        Initialize loss and metric functions.
        Note: These parameters are designed to be tuned in the appropriate yml file
        """
        logging.warning("self.datasetClass may not have the NUM_CLASSES attribute! Be sure the class specifies it")
        self.loss_f = nn.BCEWithLogitsLoss(reduction=self.run_config["LOSS_REDUCTION"]) if \
            self.datasetClass.MULTILABEL else nn.CrossEntropyLoss(reduction=self.run_config["LOSS_REDUCTION"])

        self.metrics_f = tm.MetricCollection([
            tm.Accuracy(num_classes=self.datasetClass.NUM_CLASSES, subset_accuracy=self.run_config["SUBSET_ACCURACY"]),
            tm.Precision(num_classes=self.datasetClass.NUM_CLASSES, average=self.run_config["METRIC_AVERAGE"]),
            tm.Recall(num_classes=self.datasetClass.NUM_CLASSES, average=self.run_config["METRIC_AVERAGE"]),
            tm.F1(num_classes=self.datasetClass.NUM_CLASSES, average=self.run_config["METRIC_AVERAGE"])])

    def start_training(self):
        self.run_config["TENSORBOARD_NAME"] = self.run_config["TENSORBOARD_NAME"] \
            if self.run_config["TENSORBOARD_NAME"] != "" else None
        w_decay = self.run_config.get("WEIGHT_DECAY", None)
        optimizer_params = dict(lr=self.run_config["LEARNING_RATE"])
        if w_decay is not None:
            optimizer_params["weight_decay"] = w_decay

        self.train_config = {
            "device": self.run_config["DEVICE"],
            "epochs": self.run_config["EPOCHS"],
            "loss": self.loss_f,
            "log_every_batches": 10,
            "evaluate_every": 1,
            "early_stopping": self.run_config["ES_PARAMS"],
            "optimizer": torch.optim.AdamW(self.network.parameters(), **optimizer_params),
            "path_to_best_model": self.checkpoint_path,
            "evaluator": Evaluator(self.run_config["DEVICE"], self.metrics_f,
                                   {"tensorboard_eval": self.run_config["TENSORBOARD_EVAL"],
                                    "tensorboard_name": self.run_config["TENSORBOARD_NAME"]}),
            "reload": self.checkpoint_path if self.run_config["RELOAD"] else None,
            "tensorboard_train": self.run_config["TENSORBOARD_TRAIN"],
            "tensorboard_name": self.run_config["TENSORBOARD_NAME"]
        }

        trainer = Trainer(self.network, self.train_config)
        _, score = trainer.train(self.train_loader, self.eval_loader)
        p_metrics = pd.DataFrame(score)
        print("\n*** Training metrics ***")
        print(p_metrics)

    def start_testing(self):
        evaluator = Evaluator(self.run_config["DEVICE"], self.metrics_f,
                              {"tensorboard_eval": self.run_config["TENSORBOARD_EVAL"],
                               "tensorboard_name": self.run_config["TENSORBOARD_NAME"]})
        results = evaluator.evaluate(self.test_loader, model=self.network,
                                     loss_fun=lambda b: Trainer.loss_compute_static(b, self.loss_f, self.network),
                                     path_to_model=self.checkpoint_path)

        pp = pd.Series(results).to_string(dtype=False)
        print("\n*** Test metrics ***")
        print(pp)

    def run(self):
        self.initialize_configuration()
        if self.run_config["TRAIN"]:
            if self.__warning_run_overwrite and not self.run_config["RELOAD"]:
                logging.error(
                    f"Path to checkpoints folder already exists: this probably happens because you forgot to change "
                    f"the 'RUN_NAME' parameter in 'run.yml'. To continue, either change it, or manually delete the "
                    f"existing folder: '{os.path.dirname(self.checkpoint_path)}'. Abort.")
                exit(0)
            self.initialize_train_dataset()
            if self.network is None:
                self.init_model()
            self.start_training()
        if self.run_config["TEST"]:
            self.initialize_test_dataset()
            if self.network is None:
                self.init_model()
            self.start_testing()


class TransformerScript(ScriptHandler):
    def __init__(self, tokenizer_class: Union[Type[transformers.PreTrainedTokenizer], Type[transformers.AutoTokenizer]],
                 config_class: Union[Type[transformers.PretrainedConfig], Type[transformers.AutoConfig]],
                 clf_class: Type[BaseModel],
                 dataset_class: Type[BaseDataset],
                 run_config: Union[str, Dict] = os.path.join("configs", "transformers_run.yml"),
                 model_config: Union[str, Dict] = os.path.join("configs", "transformers_models.yml")):
        super().__init__(dataset_class, run_config, model_config)
        self.pretrained_model: str = self.model_config["AVAILABLE_MODELS"][self.model_config["SELECTED_MODEL"]]
        print(f"[INFO] Running with pretrained model {self.pretrained_model}...")
        print(f"[INFO] Language: {dataset_class.LANG.title()}.")
        self.tokenizer = tokenizer_class.from_pretrained(self.pretrained_model)
        self.configClass = config_class
        self.classifierClass = clf_class

    def initialize_test_dataset(self) -> None:
        # Map-style dataset
        if self.run_config["MAP_STYLE"]:
            test_data = TransformerMapDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "test.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                self.tokenizer)

            self.test_loader = td.DataLoader(test_data, batch_size=self.run_config["TEST_BATCH_SIZE"], shuffle=False)
        # Iterable dataset
        else:
            test_data = TransformerIterableDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "test.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                batch_size=self.run_config["TEST_BATCH_SIZE"])

            self.test_loader = td.DataLoader(test_data, batch_size=None, shuffle=False,
                                             collate_fn=
                                             lambda _d: TransformerIterableDataset.collate_function(self.tokenizer, _d))

    def initialize_train_dataset(self) -> None:
        # Map-style dataset
        if self.run_config["MAP_STYLE"]:
            logging.warning("USING MAP STYLE DATASET - COULD GO OUT OF MEMORY")
            train_data = TransformerMapDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "train.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                self.tokenizer)
            eval_data = TransformerMapDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "val.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                self.tokenizer)
            # Create torch loaders
            self.train_loader = td.DataLoader(train_data, batch_size=self.run_config["TRAIN_BATCH_SIZE"], shuffle=False)
            self.eval_loader = td.DataLoader(eval_data, batch_size=self.run_config["TEST_BATCH_SIZE"], shuffle=False)
        # Iterable dataset
        else:
            train_data = TransformerIterableDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "train.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                batch_size=self.run_config["TRAIN_BATCH_SIZE"])
            eval_data = TransformerIterableDataset(
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "val.txt"),
                os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "encoder.bin.xz"),
                batch_size=self.run_config["TEST_BATCH_SIZE"])
            # Create torch loaders
            self.train_loader = td.DataLoader(train_data, batch_size=None, shuffle=False,
                                              collate_fn=lambda _d:
                                              TransformerIterableDataset.collate_function(self.tokenizer, _d))
            self.eval_loader = td.DataLoader(eval_data, batch_size=None, shuffle=False,
                                             collate_fn=lambda _d:
                                             TransformerIterableDataset.collate_function(self.tokenizer, _d))

    def init_model(self) -> None:
        """
        Initialize the transformer model
        """
        self.net_config = self.configClass.from_pretrained(self.pretrained_model,
                                                           num_labels=self.datasetClass.NUM_CLASSES)
        self.network = self.classifierClass(conf=self.net_config, **self.model_config["CLASSIFIER_CONF"],
                                            pretrained=self.pretrained_model,
                                            num_classes=self.datasetClass.NUM_CLASSES,
                                            multilabel=self.datasetClass.MULTILABEL)


class EmbeddingScript(ScriptHandler):
    def __init__(self, clf_class: Type[BaseModel], dataset_class: Type[BaseDataset], run_config: Union[str, Dict],
                 model_config: Union[str, Dict]):
        super().__init__(dataset_class, run_config, model_config)
        self.classifierClass = clf_class

        max_v = self.model_config.get("MAX_VECTORS", None)
        self.vectors, self.vocab = load_vectors(self.model_config["EMBEDDINGS"][dataset_class.LANG], max_vectors=max_v)
        self.model_config["CLASSIFIER_CONF"]["words_dim"] = self.vectors.get_vecs_by_tokens("").shape[0]
        self.model_config["CLASSIFIER_CONF"]["words_num"] = len(self.vocab)
        self.model_config["CLASSIFIER_CONF"]["num_classes"] = dataset_class.NUM_CLASSES

        self.tokenizer = Tokenizer(mod.WordLevel(self.vocab, unk_token="<unk>"))
        # normalizer = norm.Sequence([norm.NFKC(), norm.StripAccents(), norm.Lowercase()])
        self.tokenizer.normalizer = norm.BertNormalizer(strip_accents=True, lowercase=True)
        self.tokenizer.pre_tokenizer = pretok.Whitespace()
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)
        self.coll_fn = lambda b: iter_coll_fn_embed(self.tokenizer, b)

    def initialize_test_dataset(self) -> None:
        kwargs = dict(
            dataset_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "test.txt"),
            encoder_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]),
                                      "encoder.bin.xz"))
        bs = self.run_config["TEST_BATCH_SIZE"]

        # Map-style dataset
        if self.run_config["MAP_STYLE"]:
            test_data = WordEmbeddingMapDataset(**kwargs, tokenizer=self.tokenizer)
            self.test_loader = td.DataLoader(test_data, batch_size=bs, shuffle=False)
        else:
            test_data = TransformerIterableDataset(**kwargs, batch_size=bs)
            self.test_loader = td.DataLoader(test_data, batch_size=None, shuffle=False, collate_fn=self.coll_fn)

    def initialize_train_dataset(self) -> None:
        tr_kwargs = dict(
            dataset_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "train.txt"),
            encoder_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]),
                                      "encoder.bin.xz"))
        te_kwargs = dict(
            dataset_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]), "val.txt"),
            encoder_path=os.path.join(self.datasetClass.generated_root(self.run_config["DATASET_NAME"]),
                                      "encoder.bin.xz"))
        te_bs = self.run_config["TEST_BATCH_SIZE"]
        tr_bs = self.run_config["TRAIN_BATCH_SIZE"]

        # Map-style dataset
        if self.run_config["MAP_STYLE"]:
            train_data = WordEmbeddingMapDataset(**tr_kwargs, tokenizer=self.tokenizer)
            eval_data = WordEmbeddingMapDataset(**te_kwargs, tokenizer=self.tokenizer)
            self.train_loader = td.DataLoader(train_data, batch_size=tr_bs, shuffle=True)
            self.eval_loader = td.DataLoader(eval_data, batch_size=te_bs, shuffle=False)
        else:
            train_data = TransformerIterableDataset(**tr_kwargs, batch_size=tr_bs)
            eval_data = TransformerIterableDataset(**te_kwargs, batch_size=te_bs)
            self.train_loader = td.DataLoader(train_data, batch_size=None, shuffle=False, collate_fn=self.coll_fn)
            self.eval_loader = td.DataLoader(eval_data, batch_size=None, shuffle=False, collate_fn=self.coll_fn)

    def init_model(self) -> None:
        """
        Initialize the transformer model
        """
        self.network = self.classifierClass(**self.model_config["CLASSIFIER_CONF"], vectors=self.vectors.vectors,
                                            multilabel=self.datasetClass.MULTILABEL)
