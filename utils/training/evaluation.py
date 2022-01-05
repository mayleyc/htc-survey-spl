from typing import Dict, Callable, Any

import torch
import torch.utils.data as td
import torchmetrics as tm
from tqdm import tqdm

from utils.training.model import ClassificationModel
from utils.training.visualization import TensorBoard


class Evaluator:
    def __init__(self, device: str, metrics: tm.MetricCollection, eval_params: Dict[str, Any]):
        """
        :param device: the device which to run on (gpu or cpu)
        """
        self.__device = device
        self.__metrics = metrics.to(device)
        # Initialize tensorboard writer
        if eval_params["tensorboard_eval"]:
            self.board = TensorBoard(log_path='data/tensorboards/eval',
                                     log_name=eval_params['tensorboard_name'])
        else:
            self.board = None

    @staticmethod
    def format_metrics(m: Dict[str, torch.Tensor], prec: int = 4) -> str:
        """
        Pretty printer for metrics

        :param m: torchmetrics metric dictionary
        :param prec: floating point precision to use when formatting metric values
        :return: metrics for printing in a formatted string
        """
        return " | ".join(["{:s}: {:.{p}f}".format(m, v.cpu().item(), p=prec) for m, v in m.items()])

    def evaluate(self, data: td.DataLoader, model: ClassificationModel, loss_fun: Callable[[Any], torch.Tensor],
                 path_to_model: str = "") -> Dict[str, float]:
        """
        Evaluates the saved best model against train, val and test data

        :param data: evaluation data loader
        :param model: the model to evaluate
        :param loss_fun: function to be used to compute loss
        :param path_to_model: if a path to a saved model is passes, it will be used for evaluation, and `model`
            param is ignored
        """
        model = model.to(self.__device)
        model.device = self.__device
        model.eval()
        self.__metrics.reset()

        if path_to_model != "":
            state = torch.load(path_to_model, map_location=self.__device)
            if isinstance(state, dict) and state.get("model", False):
                state = state["model"]
            model.load_state_dict(state)

        # Epoch losses
        losses = list()

        with tqdm(data, total=len(data), unit="batch") as tk0:
            tk0.set_description_str("Evaluation")
            # Evaluation loop
            with torch.no_grad():
                for i, batch in enumerate(data):
                    tk0.update(1)
                    out = model(batch)
                    eval_args = model.filter_evaluate_fun_args(out)
                    loss: torch.Tensor = loss_fun(out)
                    metrics = self.__metrics(**eval_args) if isinstance(eval_args, dict) else self.__metrics(*eval_args)
                    losses.append(loss)
                    metrics = {
                        "Loss": loss,
                        **metrics
                    }
                    print_metrics: str = Evaluator.format_metrics(metrics)
                    progress: str = " Batch: {}/{} -> ### {}".format(i + 1, tk0.total, print_metrics)
                    tk0.set_postfix_str(progress)

            final_metrics = {
                "Loss": torch.mean(torch.as_tensor(losses).to(self.__device)),
                **self.__metrics.compute()
            }
            final_stats: str = " Final metrics -> ### {}".format(Evaluator.format_metrics(final_metrics))
            tk0.set_postfix_str(final_stats)
            tk0.close()

        return {k: v.cpu().item() for k, v in final_metrics.items()}

    # @staticmethod
    # def batch_accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
    #     """
    #     Computes the accuracy of the preds over the items in a single batch
    #     :param o: the logit output of datum in the batch
    #     :param y: the correct class index of each datum
    #     :return the percentage of correct preds as a value in [0,1]
    #     """
    #     corrects = torch.sum(torch.max(o, dim=1).indices.view(y.size()).data == y.data)
    #     accuracy = corrects / y.size(0)
    #     return accuracy.item()

    def batch_metrics(self, *args, **kwargs):
        metrics = {k: v.cpu().detach() for k, v in self.__metrics(*args, **kwargs).items()}
        # Write to tensorboard (batch)
        if self.board:
            self.board.write_batch_metrics(metrics, name='Eval')
        return metrics

    def epoch_metrics(self, epoch: int) -> Dict:
        metrics = {k: v.cpu().detach() for k, v in self.__metrics.compute().items()}
        # Write to tensorboard (epoch)
        if self.board:
            self.board.write_epoch_metrics(epoch, metrics, name='Eval')
        return metrics

    def reset_metrics(self) -> None:
        self.__metrics.reset()

    # @staticmethod
    # def compute_metrics(y_true, y_pred) -> Tuple[float, float, float, float, float]:
    #     acc_bal = balanced_accuracy_score(y_true, y_pred)
    #     acc = accuracy_score(y_true, y_pred)
    #     # TODO: macro average if one of them is 0 is not properly defined
    #     prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    #     return acc_bal, acc, prec_ma, rec_ma, f1_ma
    #
    # @staticmethod
    # def avg_metrics(m: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    #     pprint(m)
    #     return {kk: {k: np.mean([a[kk][k] for a in m]) for k in m[0][kk].keys()} for kk in m[0].keys()}

    # @staticmethod
    # def __optimal_roc_threshold(y_true: np.ndarray, y_1_scores: np.ndarray) -> float:
    #     """
    #     Computes the optimal ROC threshold (defined for more than one sample)
    #     :param y_true: the ground truth
    #     :param y_1_scores: the scores for the pos class
    #     :return: the optimal ROC threshold
    #     """
    #     fp_rates, tp_rates, thresholds = roc_curve(y_true, y_1_scores)
    #     best_threshold, dist = 0.5, 100
    #
    #     for i, threshold in enumerate(thresholds):
    #         current_dist = np.sqrt((np.power(1 - tp_rates[i], 2)) + (np.power(fp_rates[i], 2)))
    #         if current_dist <= dist:
    #             best_threshold, dist = threshold, current_dist
    #
    #     return best_threshold
