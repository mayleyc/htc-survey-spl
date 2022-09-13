import torch


def ce_loss(prediction, target, *args, **kwargs) -> torch.Tensor:
    kw = dict()
    device = kwargs["device"]
    if "weight" in kwargs:
        kw = dict(weight=kwargs["weight"].to(device))
    return F.cross_entropy(prediction.float().to(device), target.to(device), **kw)