import os

from src.datasets.generation.nc import RCV2it, RCV2fr, RCV1en
from src.datasets.generation.tl import ItWiki, EnWiki, FrWiki
from src.models.xmlcnn import XmlCNN
from utils.run_handler import EmbeddingScript

from utils.general import load_yaml
dataset = [ItWiki, EnWiki, FrWiki, RCV2it, RCV1en, RCV2fr][0]

if __name__ == "__main__":
    print(f"Current language: {dataset.LANG.title()}")
    run_config = load_yaml('configs/xml_run.yml')
    print(f"Current dataset: {run_config['DATASET_NAME']}")
    classification_script = EmbeddingScript(XmlCNN, dataset, run_config=os.path.join("configs", "xml_run.yml"),
                                            model_config=os.path.join("configs", "xml_model.yml"))
    classification_script.run()
