from pathlib import Path
from typing import Dict, Any, Tuple, Iterable

import torch
from torch import nn

from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.utils.torch_train_eval.generic_functions import load_model_from_path, load_model_weights
from src.utils.torch_train_eval.model import TrainableModel


class EnsembleBERT(nn.Module, TrainableModel):
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


class E2EbiBERT(nn.Module, TrainableModel):
    """
    E2E model, that train BERT with two objectives, one for each level of labels
    """

    def __init__(self, **config):
        super().__init__()
        self.__initial_config = config

        class_n_l1 = config["n_class_l1"]
        class_n_l2 = config["n_class"]

        # L2 model: This model will predict l2 (flat) labels
        self.model_l2 = BERTForClassification(**config)

        # Dimensionality of a document embedding
        supp_output_dim = self.model_l2.clf.in_features
        # L1 model: use same embeddings as L1, but predicts the l1 label
        self.model_l1 = nn.Linear(supp_output_dim, class_n_l1)

        # Create the final classifier
        self.clf = nn.Linear(class_n_l2 + class_n_l1, class_n_l2)

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor, torch.LongTensor], *args, **kwargs) -> Any:
        inp, label_l1, label_l2 = batch

        # Pass of sublabels
        l2_logits, _, doc_dropout_1_emb, doc_1_emb = self.model_l2((inp, label_l2))
        # Pass same embeddings, without creating them again
        l1_logits = self.model_l1(doc_dropout_1_emb)
        # Final classifier
        logits = self.clf(torch.cat([l1_logits, l2_logits], dim=1))

        if kwargs.get("inference", None) is None:
            return logits, l1_logits, l2_logits, label_l1, label_l2
        else:
            return logits, label_l2

    def filter_data_loss(self, outputs: Tuple, *args, **kwargs) -> Any:
        o, o1, o2, y1, y2 = outputs
        y1 = y1.long().to(self.device)
        y2 = y2.long().to(self.device)
        return o, o1, o2, y1, y2

    def constructor_args(self) -> Dict:
        return self.__initial_config

    def submodules(self) -> Iterable[Any]:
        return [self.model_l1, self.model_l2, self.clf]
