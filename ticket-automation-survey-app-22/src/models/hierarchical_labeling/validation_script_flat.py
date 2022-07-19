from src.models.hierarchical_labeling.modules.bert_classifier import BERTForClassification
from src.models.hierarchical_labeling.training_bert4c import run

if __name__ == "__main__":

    tests = ["bert_config_val_1.yml", "bert_config_val_2.yml", "bert_config_val_3.yml",
             "bert_config_val_4.yml", "bert_config_val_5.yml", "bert_config_val_6.yml",
             "bert_config_val_7.yml", "bert_config_val_8.yml", "bert_config_val_9.yml",
             "bert_config_val_10.yml", "bert_config_val_11.yml", "bert_config_val_12.yml",
             "bert_config_val_13.yml", "bert_config_val_14.yml", "bert_config_val_15.yml",
             "bert_config_val_16.yml"]

    mod_cls = BERTForClassification

    for name in tests:
        run(name, mod_cls)
