import abc
from typing import Any

from torch import nn


class BaseModel(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = None

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Take a batch of samples and make the forward pass.
        Every nn.Module subclass must implement the forward method

        :param args: if used with Trainer a batch of samples is passed as dictionary,
            but in general can take any parameter
        :param kwargs: any argument
        """
        pass

    @abc.abstractmethod
    def filter_loss_fun_args(self, *args, **kwargs) -> Any:
        """
        Give input arguments for loss function.
        Input arguments are anything returned from `predict` method

        :param args:
        :param kwargs:
        :return: dict or tuple of arguments to be passed to loss functions (they will be unpacked)
        """
        pass

    @abc.abstractmethod
    def filter_evaluate_fun_args(self, *args, **kwargs) -> Any:
        """
        Give input arguments for evaluation function.
        Input arguments are anything returned from `predict` method

        :param args:
        :param kwargs:
        :return: dict or tuple of arguments to be passed to the evaluator (they will be unpacked)
        """
        pass


class ClassificationModel(nn.Module, BaseModel):
    def __init__(self, **config):
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        self._multilabel: bool = config["multilabel"]
