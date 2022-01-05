import logging
import os
from typing import Dict, Tuple, Callable, List, Any, Optional

import numpy as np
import torch
import torch.utils.data as td
from torchinfo import summary
from tqdm import tqdm

from utils.training.evaluation import Evaluator
from utils.training.model import ClassificationModel
from utils.training.visualization import TensorBoard


class Trainer:
    def __init__(self, model: ClassificationModel, train_params: Dict[str, Any]):
        """
        Build a trainer for a model

        :param model: model to be trained
        :param train_params: the train related params in the experiment.json file
            - path_to_best_model
            - device
            - epochs
            - loss
            - log_every_batches
            - evaluate_every
            - optimizer
            - evaluator
            - early_stopping : { patience, metrics, metrics_trend }
            - reload
        """
        self.__path_to_best_model = train_params["path_to_best_model"]

        self.__device = train_params["device"]
        self.__epochs = train_params["epochs"]
        self.__start_epoch = 0
        self.__loss: Optional[Callable] = train_params.get("loss", None)

        self.__log_every = train_params.get("log_every_batches", 1)
        self.__evaluate_every = train_params["evaluate_every"]

        self.__patience = train_params["early_stopping"]["patience"]
        self.__es_metric = train_params["early_stopping"]["metrics"]
        self.__es_metric_trend = train_params["early_stopping"]["metrics_trend"]
        self.__es_metric_best_value = 0.0 if self.__es_metric_trend == "increasing" else 1000
        self.__epochs_no_improvement = 0

        self.model = model.to(self.__device)
        self.model.device = self.__device
        self.__optimizer = train_params["optimizer"]

        self.__evaluator: Evaluator = train_params["evaluator"]
        # Initialize tensorboard writer
        if train_params["tensorboard_train"]:
            self.board = TensorBoard(log_path='data/tensorboards/train',
                                     log_name=train_params['tensorboard_name'])
        else:
            self.board = None

        reload = train_params.get("reload", None)
        if reload:
            self.load_state()
            print("Loaded model state from {:s}".format(reload))

    def train(self, train_data: td.DataLoader, val_data: td.DataLoader) -> \
            Tuple[ClassificationModel, Dict[str, Dict[str, float]]]:
        """
        Trains the model according to the established parameters and the given data

        :param train_data: training dataloader
        :param val_data: validation dataloader
        :return: the evaluation metrics of the training and the trained model
        """
        summary(self.model)

        print("\n Training the model... \n")

        # Evaluate every CV split
        evaluations: List[Dict[str, Dict[str, float]]] = list()

        num_epochs: int = self.__start_epoch + self.__epochs
        for epoch in range(self.__start_epoch, num_epochs):
            print("\n *** Epoch {}/{} *** ".format(epoch + 1, num_epochs))

            # Set training mode
            self.model.train()
            self.__evaluator.reset_metrics()

            # Keep track of loss for each batch
            losses = list()

            with tqdm(train_data, total=len(train_data), unit="batch") as tk0:
                tk0.set_description_str("Training  ")  # Spaces to pad it to 10 letters

                status: str = ""
                for i, batch in enumerate(train_data):
                    tk0.update(1)
                    # Reset gradient
                    self.__optimizer.zero_grad()
                    # Forward step
                    out = self.model(batch, e=epoch, b=i)
                    # Get arguments for metrics computing
                    eval_args = self.model.filter_evaluate_fun_args(out)

                    # Compute training loss and metrics
                    loss = self.training_loss(out)
                    running_metrics = self.__evaluator.batch_metrics(**eval_args) if isinstance(eval_args, dict) else \
                        self.__evaluator.batch_metrics(*eval_args)

                    loss.backward()
                    self.__optimizer.step()

                    # Save loss for batch
                    losses.append(loss.cpu().detach().item())

                    if not (i + 1) % self.__log_every:
                        updated_metrics = {
                            "Loss": torch.mean(torch.as_tensor(losses)),
                            **running_metrics
                        }
                        print_metrics: str = Evaluator.format_metrics(updated_metrics)
                        status = "-> ### {:s} ".format(print_metrics)

                        # Write to tensorboard
                        if self.board:
                            self.board.write_batch_metrics(updated_metrics)

                    progress: str = " Batch: {}/{} {}".format(i + 1, len(train_data), status)
                    tk0.set_postfix_str(progress)

                # Compute final metrics and loss of epoch

                final_metrics = {
                    "Loss": torch.mean(torch.as_tensor(losses)),
                    **self.__evaluator.epoch_metrics(epoch)
                }
                final_stats: str = " Final metrics -> ### {:s}".format(Evaluator.format_metrics(final_metrics))
                tk0.set_postfix_str(final_stats)
                tk0.close()

                # Write to tensorboard
                if self.board:
                    self.board.write_epoch_metrics(epoch, final_metrics)

            # Validation step and early stopping check

            if (epoch + 1) % self.__evaluate_every == 0:
                epoch_metrics = {
                    "val": self.__evaluator.evaluate(val_data, self.model, loss_fun=self.evaluation_loss),
                    "train": {k: v.cpu().item() for k, v in final_metrics.items()}
                }
                evaluations.append(epoch_metrics)
                if self.__early_stopping_check(evaluations[-1]["val"], epoch + 1):
                    break

        # Return metrics of best model (the saved one)
        ms = [m["val"][self.__es_metric] for m in evaluations]
        best_idx = np.argmax(ms) if self.__es_metric_trend == "increasing" else np.argmin(ms)
        best_metrics = evaluations[best_idx]

        print("\n Finished training! \n")
        # Make sure that all pending events have been written to disk
        if self.board:
            self.board.flush()
            self.__evaluator.board.flush()

        return self.model, best_metrics

    def __early_stopping_check(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """
        Decides whether or not to early stop the train based on the early stopping conditions

        :param val_metrics: the monitored val metrics (e.g. auc, loss)
        :param epoch: number of last completed epoch (starting from 1 for the first one)
        :return: a flag indicating whether or not the training should be early stopped
        """
        metric_value = val_metrics[self.__es_metric]
        if self.__es_metric_trend == "increasing":
            metrics_check = metric_value > self.__es_metric_best_value
        else:
            metrics_check = metric_value < self.__es_metric_best_value

        # Override metrics check for first epoch
        if epoch == 1:
            metrics_check = True

        if metrics_check:
            self.save_state(epoch, val_metrics)
            print("\t Old best val {m}: {o:.4f} | New best {m}: {n:.4f} -> New best model saved!"
                  .format(m=self.__es_metric, o=self.__es_metric_best_value, n=metric_value))

            self.__es_metric_best_value = metric_value
            self.__epochs_no_improvement = 0
        else:
            self.__epochs_no_improvement += 1
            if self.__epochs_no_improvement == self.__patience:
                print("\t ** No decrease in val {} for {} evaluations. Early stopping! ** \n"
                      .format(self.__es_metric, self.__patience, self.__evaluate_every))
                return True
            else:
                print("\t No improvement. Epochs without improvement: ", self.__epochs_no_improvement)
        return False

    def save_state(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save the model and optimizer states on file, overwriting the file if it exists
        """
        # Rename old best checkpoint file
        file, ext = os.path.splitext(self.__path_to_best_model)
        new_name = f"{file}_e{epoch - 1}_{self.__es_metric_best_value:.4f}_{self.__es_metric}{ext}"
        if os.path.isfile(new_name):
            os.remove(new_name)
        if os.path.isfile(self.__path_to_best_model):
            os.rename(self.__path_to_best_model, new_name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.__optimizer.state_dict(),
            "epochs": epoch,
            "metrics": metrics
        }, self.__path_to_best_model)

    def load_state(self) -> None:
        """
        Load a model and optimizer states from file
        """
        print("Reloading checkpoint {:s}...".format(self.__path_to_best_model))
        state = torch.load(self.__path_to_best_model, map_location=self.__device)
        self.model.load_state_dict(state["model"])
        self.__optimizer.load_state_dict(state["optimizer"])
        self.__start_epoch = state["epochs"]
        m = state.get("metrics", None)
        if m:
            self.__es_metric_best_value = m[self.__es_metric]
        else:
            logging.warning("Training is resuming but no metric best value is saved in checkpoint. "
                            "It will be reinitialized.")

    # --------------------- Overridable methods ---------------------

    def training_loss(self, out: Any) -> torch.Tensor:
        """
        Compute training error (loss).
        Default behaviour calls `filter_loss_fun_args` with the model output and pass results as arguments to the
        loss function passed as a parameter to the trainer `__init__` method.
        You can redefine this method if you need custom loss computation.

        :param out: the output of the `predict` method of the model
        :return: by default it computes loss using the callable parameter supplied to the trainer. Must return a tensor.
        """
        return Trainer.loss_compute_static(out, self.__loss, self.model)

    def evaluation_loss(self, out: Any) -> torch.Tensor:
        """
        Compute loss for validation/test data.
        Defaults to `training_loss`. If something different is required, override this method.

        :param out: the output of `predict` method of the model
        :return: the tensor with loss values for input
        """
        return self.training_loss(out)

    @staticmethod
    def loss_compute_static(out: Any, loss_fun: Callable, model: ClassificationModel) -> torch.Tensor:
        """
        A wrapper for default loss computation. Can be used by any external script for evaluation.

        :param out: output of the `predict` method
        :param loss_fun: callable object to use for loss computation
        :param model: model to be used to get loss arguments
        :return: a tensor with loss function value
        """
        loss_args = model.filter_loss_fun_args(out)
        return loss_fun(**loss_args) if isinstance(loss_args, dict) else loss_fun(*loss_args)
