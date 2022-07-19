from src.models.hierarchical_labeling.modules.ensembles import E2EbiBERT
from src.models.hierarchical_labeling.training_bert4c import run

EPOCH = 3


def train_e2e():
    file_name = "e2e_2l_config.yml"
    run(file_name, E2EbiBERT, split_fun=lambda n: dict())


if __name__ == "__main__":
    train_e2e()
