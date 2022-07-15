import logging
import os

from experiments.fast_text import main
from utils.general import load_yaml

folders = ["rcv1en-4", "rcv2fr-4", "rcv2it-4", "enwiki-4", "itwiki-4", "frwiki-4"]

if __name__ == "__main__":
    logging.basicConfig(filename=os.path.join("experiments", "ft_results.log"), level=logging.INFO)

    conf = load_yaml(os.path.join("configs", "ft_run.yml"))
    for ds in folders:
        conf["DATASET_NAME"] = ds
        main(conf)
