from pathlib import Path

# matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.analysis.seabornfig2grid import SeabornFig2Grid
from src.dataset_tools.prepare_linux_dataset import read_dataset as read_bugs
from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.models.hierarchical_labeling.modules.utility_functions import _setup_training
from src.utils.torch_train_eval.generic_functions import load_yaml

device = "cuda"

colors = [
    "#47A5BF",
    "#4BC9C3",
    "#4BB390",
    "#4BC97B",
    "#47BF53",
    "#91CC41",
    "#D8E33D",
    "#D9CC3B",
    "#F0CD35",
    "#E6B232",
    "#FCAC2B",
    "#F28B29",
    "#DB5B1A",
    "#F9481E"
]


def analyze(model, loader, plot_words_distribution: bool = True):
    model.train(False)
    with torch.no_grad():
        for i, pred_data in tqdm(enumerate(loader), total=len(loader)):
            sns.set_style("darkgrid")
            batch, labels = pred_data
            input_ids = batch["input_ids"].long().to(device)
            attention_mask = batch["attention_mask"].float().to(device)
            bert_outputs = model.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            pooled_output, last_states, hidden_states = bert_outputs.pooler_output, bert_outputs.last_hidden_state, bert_outputs.hidden_states

            # With linear and tanh
            # pooled = pooled_output.detach().cpu().numpy().squeeze()
            # Before last
            hidden_squeezed = []
            for state in hidden_states:
                x = state.detach().cpu().numpy()  # (1, seq_len, 768)
                if plot_words_distribution:
                    x = x.squeeze().flatten()
                else:
                    x = x.transpose(2, 0, 1).reshape(x.shape[2], -1)  # (768, word num in batch)
                hidden_squeezed.append(x)

            fig = plt.figure(figsize=(13, 8))
            fig.suptitle(f"Hidden state values throughout the model", fontsize=30)
            gs = gridspec.GridSpec(4, 4)
            mgs = []
            for i, state in enumerate(hidden_squeezed):
                g = sns.jointplot(data=state, color=colors[i],
                                  marker="+", marginal_ticks=True, palette="inferno")
                g.ax_marg_x.set_title(f"Output of hidden layer {i}")
                # g.ax_joint.set_xscale("log")
                g.ax_joint.set_yscale("log")
                s = SeabornFig2Grid(g, fig, gs[i])
                mgs.append(s)

            # gs.tight_layout(fig)
            plt.show()


def main():
    df = read_bugs()

    mod_cls = BERTForClassification
    workers = 0
    config_path = Path("src") / "models" / "hierarchical_labeling" / "configs" / "bert_config_analysis.yml"
    config = load_yaml(config_path)

    tickets = df["message"]
    labels = df[config["LABEL"]]
    config["n_class"] = 42

    x_train, x_test, y_train, y_test = train_test_split(tickets, labels, test_size=0.2, random_state=936818217,
                                                        stratify=labels)

    trainer, train_load, val_load = _setup_training(train_config=config, model_class=mod_cls,
                                                    workers=workers,
                                                    data=x_train, labels=y_train,
                                                    data_val=x_test, labels_val=y_test)
    # TEST the model
    model = trainer.model
    # Use the model to predict test/validation samples
    analyze(model, train_load, plot_words_distribution=True)  # (samples, num_classes)


if __name__ == "__main__":
    main()
