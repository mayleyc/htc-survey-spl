import pandas as pd

from src.dataset_tools.data_preparation.prepare_linux_dataset import read_dataset as read_bugs
from nltk import word_tokenize
from transformers import AutoTokenizer


def run_analysis_bugs():
    df: pd.DataFrame = read_bugs(verbose=True)
    print("----------------------------------------------------------------------------------")
    print("Applying standard tokenization...")
    standard_tokenization = df['message'].apply(word_tokenize)
    print(f"Average number of tokens (word_tokenize): {standard_tokenization.apply(len).mean()}")
    print("----------------------------------------------------------------------------------")
    print("Applying BERT subword tokenization...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tokenization = df['message'].apply(bert_tokenizer.encode)
    print(f"Average number of tokens (bert-base-uncased tokenizer): {bert_tokenization.apply(len).mean()}")
    print("----------------------------------------------------------------------------------")


def run_analysis_wos():
    pass


def run_analysis_rcv1():
    pass


def run_analysis_bgc():
    pass


if __name__ == "__main__":
    run_analysis_bugs()
