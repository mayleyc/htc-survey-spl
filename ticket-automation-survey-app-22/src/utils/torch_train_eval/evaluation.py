import collections
from typing import Dict, Callable, Tuple, Union

import torch
import torchmetrics as tm


class MetricSet(collections.UserDict):
    """
    Wrapper around torchmetrics Metric and MetricCollection objects.
    It adds the ability to filter arguments, but can be used transparently as a Metric object.

    Initialization:
        - data: a dictionary of metrics, with a name and a tuple containing the torchmetrics metric object and a callback to filter arguments that are passed to it
        - device: the torch device where to evaluate metrics
    """

    def __init__(self, data: Dict[str, Tuple[Union[tm.Metric, tm.MetricCollection], Callable]] = dict(), device: torch.device = None):
        super().__init__(data)
        # Data is set in the superclass, this is just to document the type
        self.data: Dict[str, Tuple[Union[tm.Metric, tm.MetricCollection], Callable]]
        self._device = device

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on a batch, accumulating results.

        @param args: any positional arg
        @param kwargs: any key-value arg
        @return: dictionary with resulting metric value, over the current batch only
        """
        result = dict()
        for metric_name, (metric, filter_function) in self.items():
            metric.to(self._device)
            filtered_results = filter_function(*args, **kwargs, device=self._device)
            if isinstance(filtered_results, dict):
                result[metric_name] = metric(**filtered_results)
            else:
                result[metric_name] = metric(*filtered_results)

        self.__flatten_subdicts(result)

        return result

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics over all batches

        @return: resulting metrics over all accumulated batches
        """
        result = dict()
        for metric_name, (metric, _) in self.items():
            metric.to(self._device)
            result[metric_name] = metric.compute()

        self.__flatten_subdicts(result)
        return result

    def reset(self, metric_name: str = None) -> None:
        """
        Reset the accumulators, of one or all metrics.
        To be called after each epoch, before starting to evaluate the first batch.

        @param metric_name: if a name is passed only that metric is reset, otherwise all metrics are reset
        """
        if metric_name is not None:
            self[metric_name][0].reset()
        else:
            for metric, _ in self.values():
                metric.reset()

    @staticmethod
    def __flatten_subdicts(d: Dict) -> None:
        """
        Remove sub-dictionaries, replacing them with their key-value pairs

        @param d: dictionary to flatten in-place
        """
        sub_dicts = [(k, v) for k, v in d.items() if isinstance(v, dict)]
        if sub_dicts:
            k, v = sub_dicts[0]
            d.pop(k)
            new_values = v
            for k, v in sub_dicts[1:]:
                d.pop(k)
                new_values.update(v)
            d.update(new_values)


def format_metrics(metric_set: Dict[str, torch.Tensor], precision: int = 4) -> str:
    """
    Pretty printer for metrics

    @param metric_set: torchmetrics metric dictionary
    @param precision: floating point precision to use when formatting metric values
    @return: metrics for printing in a formatted string
    """
    return " | ".join([f"{metric:s}: {value.cpu().item():.{precision}f}"
                       for metric, value in metric_set.items()])

    # class Evaluator:
    #     def __init__(self, device: str, metrics: MetricSet,
    #                  eval_config: Dict[str, Any], board: Optional[SummaryWriter] = None):
    #         """
    #         Build an Evaluator to run on a set of metrics on a device
    #
    #         @param device: the device which to run on (gpu or cpu)
    #         @param metrics: a dict-like object containing metrics to be evaluated
    #         @param eval_config: initialization params
    #         """
    #         # Path to folder for dumps
    #         self._model_folder = Path(eval_config["MODEL_FOLDER"])
    #         # Generic parameters
    #         self.__log_every_batches = 100
    #         self.device: str = device
    #         self._metrics: MetricSet = metrics
    #         self.old_values = None
    #         # Board writer
    #         enable_board: bool = eval_config.get("TENSORBOARD", None)
    #         self.board = None
    #         if enable_board:
    #             if not board:
    #                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #                 self.board = SummaryWriter(str(self._model_folder / f"Tensorboard_{timestamp}"))
    #             else:
    #                 self.board = board

    # def evaluate_batch(self, outputs, labels, i):
    #     self._metrics.reset()
    #     metrics = self._metrics(outputs, labels)
    #
    #     print_metrics: str = Evaluator.format_metrics(metrics)
    #     if i % self.__log_every_batches == 0:
    #         return print_metrics
    #         # tqdm_bar.set_postfix_str(progress)

    # def evaluate_epoch(self, data_loader: td.DataLoader, model: nn.Module,
    #                    path_to_model: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    #     """
    #     Evaluates the saved best model against train, val and test data
    #
    #     @param data_loader: evaluation data loader
    #     @param model: the model to evaluate
    #     @param path_to_model: if a path to a saved model is passed, it will be used for evaluation,
    #                           and the `model` param will be ignored.
    #     """
    #     raise DeprecationWarning
    #     # Visual progress bar
    #     tqdm_bar = tqdm(data_loader, total=len(data_loader), unit="batch")
    #     tqdm_bar.set_description_str("Evaluation")
    #     # CPU/GPU as available
    #     model = model.to(self.device)
    #     model.device = self.device
    #     model.eval()
    #     self._metrics.reset()
    #     # Load specific model, if specified
    #     if path_to_model:
    #         model.load_state_dict(torch.load(path_to_model, map_location=self.device))
    #     # Evaluation loop
    #     with torch.no_grad():
    #         for i, data in enumerate(data_loader):
    #             tqdm_bar.update(1)
    #             # Every data instance is an input + label pair
    #             inputs, labels = data
    #             # Pass to GPU, if specified
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)
    #             # Predict
    #             outputs = model(inputs)
    #             # Eventual filtering operations for evaluation (e.g., sigmoid for logits)
    #             # outputs, labels = model.filter_data_evaluation(outputs, labels)
    #             metrics = self._metrics(outputs, labels)
    #
    #             print_metrics: str = Evaluator.format_metrics(metrics)
    #             if i % self.__log_every_batches == 0:
    #                 progress: str = f" Batch: {i + 1}/{tqdm_bar.total} {print_metrics}"
    #                 tqdm_bar.set_postfix_str(progress)
    #
    #     final_metrics = {
    #         **self._metrics.compute()
    #     }
    #
    #     # final_stats: str = f" Final metrics -> ### {Evaluator.format_metrics(final_metrics)}"
    #     # tqdm_bar.set_postfix_str(final_stats)
    #     tqdm_bar.close()
    #
    #     values = {k: v.cpu().item() for k, v in final_metrics.items()}
    #     if self.board:
    #         # tb_x = epoch_num * len(training_loader) + i + 1
    #         for k, m in values:
    #             self.board.add_scalar(f'{k}/val', m)
    #     report_last_values = self.old_values
    #     self.old_values = values
    #     return values, report_last_values

    # THESE are not used for now, since they were used to evaluate training batches
    # def batch_metrics(self, *args, **kwargs):
    #     metrics = {k: v.cpu().detach() for k, v in self.__metrics(*args, **kwargs).items()}
    #     # Write to tensorboard (batch)
    #     if self.board:
    #         self.board.write_batch_metrics(metrics, name='Eval')
    #     return metrics
    #
    # def epoch_metrics(self, epoch: int) -> Dict:
    #     metrics = {k: v.cpu().detach() for k, v in self.__metrics.compute().items()}
    #     # Write to tensorboard (epoch)
    #     if self.board:
    #         self.board.write_epoch_metrics(epoch, metrics, name='Eval')
    #     return metrics
    #
    # def reset_metrics(self) -> None:
    #     self.__metrics.reset()

    # def filter_data_evaluation(self, outputs: torch.Tensor, labels: torch.Tensor):
    #     outputs = torch.sigmoid(outputs).to(self.device)
    #     labels = labels.long().to(self.device)
    #     return outputs, labels
