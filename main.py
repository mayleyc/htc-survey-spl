from transformers import AutoTokenizer, AutoConfig

from src.datasets.generation.sa import SentiPolc
from src.datasets.generation.nc import RCV2it, RCV2fr, RCV1en
from src.datasets.generation.tl import ItWiki, EnWiki, FrWiki
from src.models.transformer import TransformerClassifier
from utils.run_handler import TransformerScript

# DESIRED DATASET MUST BE SELECTED HERE
# dataset = ItWiki
from src.models.keras_bonanza.keras_utils import get_data
import numpy as np


def run():
    # classification_script = TransformerScript(AutoTokenizer, AutoConfig, TransformerClassifier, dataset)
    # classification_script.run()
    dataset = FrWiki()
    (train_x, _,
     val_x, _,
     test_x, _) = get_data(dataset)
    tr = np.mean([len(x.split()) for x in train_x])
    va = np.mean([len(x.split()) for x in val_x])
    te = np.mean([len(x.split()) for x in test_x])
    print(f"{dataset.__class__.__name__} has an average of \n"
          f"  TRAIN      : {round(tr)}\n"
          f"  VALIDATION : {round(va)}\n"
          f"  TEST       : {round(te)}\n"
          f"  OVERALL    : {round((tr+va+te)/3)}\n"
          f"tokens")

    # rcv2_it = RCV2it()
    # rcv2_it.generate(plot=False)
    # #
    # rcv2_fr = RCV2fr()
    # rcv2_fr.generate(plot=False)
    #
    # rcv1_en = RCV1en()
    # rcv1_en.generate(plot=False)


if __name__ == "__main__":
    run()
