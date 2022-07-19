from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.models.hierarchical_labeling.training_bert4c import run

if __name__ == "__main__":

    tests = [f"bert_config_test_24.yml"]

    mod_cls = BERTForClassification

    for name in tests:
        run(name, mod_cls)
