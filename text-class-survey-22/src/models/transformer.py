from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from utils.training.model import ClassificationModel


class TransformerClassifier(ClassificationModel):
    def __init__(self, **config):
        super().__init__(**config)
        model_config = config["conf"]
        pretrained = config["pretrained"]
        freeze: bool = config.get("freeze_base", False)
        self.clf = AutoModelForSequenceClassification.from_pretrained(pretrained, config=model_config)
        if freeze:
            for param in self.clf.base_model.parameters():
                param.requires_grad = False

    def forward(self, batch: Dict[str, torch.Tensor], *args, **kwargs) -> Any:
        # TODO check kwargs if unused by trainer
        input_ids = batch["input_ids"].long().to(self.device)
        attention_mask = batch["attention_mask"].float().to(self.device)
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.long().to(self.device)
        outputs = self.clf(input_ids, attention_mask=attention_mask)
        return outputs, labels

    def filter_loss_fun_args(self, out) -> Any:
        out, y = out
        if self._multilabel:
            y = y.float().to(self.device)
        return out.logits, y

    def filter_evaluate_fun_args(self, out) -> Any:
        out, labels = out
        logits = out["logits"].float().to(self.device)
        # For single-class task use softmax, use sigmoid for multi-label
        if self._multilabel:
            probs = torch.sigmoid(logits).to(self.device)
        else:
            probs = F.softmax(logits, dim=1).to(self.device)
        return probs, labels
