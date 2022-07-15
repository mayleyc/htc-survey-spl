from transformers import AutoTokenizer, AutoConfig

from src.datasets.generation.nc import RCV2fr, RCV1en, RCV2it
from src.datasets.generation.tl import ItWiki, EnWiki, FrWiki
from src.models.transformer import TransformerClassifier
from utils.run_handler import TransformerScript

dataset = [ItWiki, EnWiki, FrWiki, RCV2it, RCV1en, RCV2fr][2]

if __name__ == "__main__":
    classification_script = TransformerScript(AutoTokenizer, AutoConfig, TransformerClassifier, dataset)
    classification_script.run()
