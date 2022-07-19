from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class DoubleCE(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        if weight is not None:
            raise ValueError("Weighted loss in unsupported when training E2EbiBERT")

    def forward(self, o, o1, o2, y1, y2) -> torch.Tensor:
        l1 = F.cross_entropy(o1, y1, reduction="mean")
        l2 = F.cross_entropy(o2, y2, reduction="mean")
        l_all = F.cross_entropy(o, y2, reduction="mean")
        return l1 + l2 + l_all
