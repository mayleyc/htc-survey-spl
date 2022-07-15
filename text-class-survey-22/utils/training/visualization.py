from typing import Dict, List, Union
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
import socket
import torch
import os


class TensorBoard:
    def __init__(self, minibatch_interval: int = 50,
                 log_path: str = 'data/tensorboards',
                 log_name: str = None):
        if not log_name:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                log_path, current_time + '_' + socket.gethostname())
        else:
            log_dir = os.path.join(
                log_path, log_name)

        self.__tb_writer: SummaryWriter = SummaryWriter(log_dir=log_dir)
        self.__batch_iter: int = 0
        self.__minibatch_interval = minibatch_interval

    def write_epoch_metrics(self, epoch: int, metrics: Dict[str, torch.Tensor],
                            name: str = 'Training') -> None:
        for metric in metrics.keys():
            self.__tb_writer.add_scalar(name + " metrics [per epoch]/" + metric, metrics[metric], epoch)

    def write_batch_metrics(self, metrics: Dict[str, torch.Tensor],
                            name: str = 'Training'):
        if self.__batch_iter % self.__minibatch_interval == 0:  # every minibatch_inteval...
            for metric in metrics.keys():
                self.__tb_writer.add_scalar(name + " metrics [per batch]/" + metric,
                                            metrics[metric], self.__batch_iter)
        self.__batch_iter += 1

    def draw_model(self, net: torch.nn.Module, input_to_model: Union[torch.Tensor, List[torch.Tensor]]):
        self.__tb_writer.add_graph(net, input_to_model)
        # idx = torch.randint(len(train_data), (1,))
        # self.board.draw_model(self.model, train_data.dataset[idx])

    # [Discarded] Pr curve for evaluator?
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # TODO: multilabel confusion matrix?
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html

    def flush(self) -> None:
        self.__tb_writer.flush()


def _confusion_matrix_multilabel(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)


def plot_cm_multilabel(y_true, y_pred, labels, subplot_rows: int, subplot_columns: int):
    # NOTE: Funziona ma non troppo; lascio come monito di non fare una confusion matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(subplot_rows, subplot_columns, figsize=(16, 16))

    for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
        _confusion_matrix_multilabel(cfs_matrix, axes, label, ["Negative", "Positive"])

    fig.tight_layout()
    plt.show()


def main():
    from sklearn.preprocessing import MultiLabelBinarizer
    print("")
    y_t = np.array([[1, 0, 3], [0, 1], [0, 2, 1], [2, 1]])
    y_p = np.array([[1, 0, 3], [1, 2], [0, 1, 3], [1, 2]])
    encoder = MultiLabelBinarizer()
    y_t = encoder.fit_transform(y_t)
    y_p = encoder.transform(y_p)
    plot_cm_multilabel(y_t, y_p, ["0", "1", "2", "3"], 2, 2)


if __name__ == "__main__":
    main()
