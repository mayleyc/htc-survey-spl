from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary

from src.utils.torch_train_eval.model import TrainableModel


class XLNetForClassification(nn.Module, TrainableModel):
    def __init__(self, **config):
        super().__init__()
        # Params
        self.__initial_config = config
        pretrained = config["PRETRAINED_LM"]
        class_n = config["n_class"]
        freeze: bool = config.get("FREEZE_BASE", False)
        if freeze:
            for param in self.lm.base_model.parameters():
                param.requires_grad = False
        # XLNet model
        self.lm = AutoModel.from_pretrained(pretrained, config=config)
        # Creates summary for output sequences
        self.sequence_summary = SequenceSummary(config)
        # Classifier
        lm_embedding_size = self.lm.config.hidden_size
        self.clf = nn.Linear(lm_embedding_size, class_n)

    def forward(self, batch: Tuple[Dict[str, torch.Tensor], torch.LongTensor], *args, **kwargs) -> Any:
        batch, labels = batch
        input_ids = batch["input_ids"].long().to(self.device)
        attention_mask = batch["attention_mask"].float().to(self.device)
        last_hidden_state = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        doc_emb = self.sequence_summary(last_hidden_state[0])

        logits = self.clf(doc_emb)
        return logits, labels, doc_emb

    def filter_data_loss(self, outputs: Tuple, *args, **kwargs) -> Any:
        o, y, *_ = outputs
        y = y.long().to(self.device)
        return o, y

    def constructor_args(self) -> Dict:
        return self.__initial_config

    def filter_evaluate_fun_args(self, out) -> Any:
        out, labels = out
        logits = out.float().to(self.device)
        # For single-class task use softmax, use sigmoid for multi-label
        probs = F.softmax(logits, dim=1).to(self.device)
        return probs, labels
