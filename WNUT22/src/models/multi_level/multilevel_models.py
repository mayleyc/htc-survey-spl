from pathlib import Path
from typing import Dict, Any, Tuple, Iterable

import torch
from torch import nn

from src.models.bert_clf.bert_classifier import BERTForClassification
from src.utils.generic_functions import load_model_from_path, load_model_weights
from src.utils.torch_train_eval.model import TrainableModel


class MultiLevelBERT(nn.Module, TrainableModel):
    """
    Flattened classification by ebsembling two pre-trained BERT models on L1 and Flat tasks respectively.
    """

    def __init__(self, **config):
        super().__init__()
        self.__initial_config = config
        class_n = config["n_class"]

        # Paths to .pt files with models' weights
        weights_dump_1 = Path(config["MODEL_L1"])
        weights_dump_2 = Path(config["MODEL_L2"])

        # Create model instances
        self.model_1 = load_model_from_path(weights_dump_1.parent.parent)
        self.model_2 = load_model_from_path(weights_dump_2.parent.parent)

        # Reload weights from checkpoint
        load_model_weights(self.model_1, weights=weights_dump_1)
        load_model_weights(self.model_2, weights=weights_dump_2)

        # Freeze weights in both base models
        for param in self.model_1.parameters():
            param.requires_grad = False
        for param in self.model_2.parameters():
            param.requires_grad = False

        clf_in_size = self.model_1.clf.in_features + self.model_2.clf.in_features
        self.clf = nn.Linear(clf_in_size, class_n)

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor], *args, **kwargs) -> Any:
        inp, labels = batch

        # First two are logits, labels, output emb with dropout, output without dropout
        with torch.no_grad():
            _, _, doc_dropout_1, doc_1 = self.model_1(batch)
            _, _, doc_dropout_2, doc_2 = self.model_2(batch)

        in_clf = torch.cat([doc_dropout_1, doc_dropout_2], dim=1).to(self.device)
        logits = self.clf(in_clf)
        return logits, labels

    def filter_data_loss(self, outputs: Tuple, *args, **kwargs) -> Any:
        o, y = outputs
        y = y.long().to(self.device)
        return o, y

    def constructor_args(self) -> Dict:
        return self.__initial_config

    def submodules(self) -> Iterable[Any]:
        return [self.model_1, self.model_2, self.clf]


class SupportedBERT(nn.Module, TrainableModel):
    """
    Train BERT on classification task on flattened labels from the input text, supported by a BERT pre-trained to predict L1 labels
    """

    def __init__(self, **config):
        super().__init__()
        self.__initial_config = config

        # Create the support model instance
        weights_dump_1 = Path(config["MODEL_L1"])
        self.model_1 = load_model_from_path(weights_dump_1.parent.parent)
        # Reload weights from checkpoint
        load_model_weights(self.model_1, weights=weights_dump_1)
        # Freeze weights in base model
        for param in self.model_1.parameters():
            param.requires_grad = False

        # BERT4C receives an additional input, which is the output of model_1
        config["additional_input"] = self.model_1.clf.in_features

        # Create the BERT supported classifier (supported by model_1)
        self.clf = BERTForClassification(**config)

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor], *args, **kwargs) -> Any:
        inp, labels = batch

        with torch.no_grad():
            _, _, doc_dropout_1, doc_1 = self.model_1(batch)
        logits, *_ = self.clf(batch, add_clf_inputs=(doc_1,))

        return logits, labels

    def filter_data_loss(self, outputs: Tuple, *args, **kwargs) -> Any:
        o, y = outputs
        y = y.long().to(self.device)
        return o, y

    def constructor_args(self) -> Dict:
        return self.__initial_config

    def submodules(self) -> Iterable[Any]:
        return [self.model_1, self.clf]
