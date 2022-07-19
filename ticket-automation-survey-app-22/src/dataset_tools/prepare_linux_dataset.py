from pathlib import Path

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from src.utils.torch_train_eval.generic_functions import simplify_text

dataset = Path("data") / "all_linux_bugs.csv.gz"


def read_dataset(threshold: int = 100) -> pd.DataFrame:
    """
    Read the dataset from file and return it cleaned of null values/rows

    :param threshold: Minimum numbers of class representatives necessary for a class to be included in the final
                      dataset
    :return: filtered data (as pd.df)
    """
    # Read all data in a DataFrame
    data = pd.read_csv(dataset, sep="\t")
    print(data.shape[0])

    # Keep only tickets that have a message description
    filtered_data = data[~data["message"].isnull()]
    print(filtered_data.shape[0])
    print(f"Average character number in ticket body: {filtered_data['message'].str.len().mean():.2f}")

    # Delete duplicate rows
    filtered_data = filtered_data.drop_duplicates(["message", "title"], keep=False)
    filtered_data = filtered_data.dropna(axis=0)

    filtered_data["message"] = filtered_data["title"].astype(str) + " " + filtered_data["message"].astype(str)

    # Find missing values in product and sub_product
    null_prods = filtered_data["product"].isnull().sum()
    null_comps = filtered_data["component"].isnull().sum()
    print(f"{'Nan values in <product>:':<28}{null_prods:>5}")
    print(f"{'Nan values in <component>:':<28}{null_comps:>5}")
    print(f"{'Nan values in <title>:':<28}{filtered_data['title'].isnull().sum():>5}")
    print(f"{'Nan values in <message>:':<28}{filtered_data['message'].isnull().sum(): >5}")

    keep = ["product", "component", "message"]
    filtered_data = filtered_data.loc[:, keep]
    filtered_data.columns = ["label", "sub_label", "message"]

    filtered_data.loc[:, "label"] = filtered_data["label"].map(simplify_text)
    filtered_data = filtered_data.dropna(axis=0)
    filtered_data.loc[:, "sub_label"] = filtered_data["sub_label"].map(simplify_text)
    filtered_data = filtered_data.dropna(axis=0)
    filtered_data["flattened_label"] = filtered_data["label"] + "_" + filtered_data["sub_label"]
    filtered_data = filtered_data.reset_index(drop=True)

    # Remove categories appearing < tot times
    c1 = filtered_data["label"].value_counts()
    filtered_data = filtered_data.replace(c1[c1 < threshold].index, np.nan).dropna(axis=0)
    c2 = filtered_data["sub_label"].value_counts()
    filtered_data = filtered_data.replace(c2[c2 < threshold].index, np.nan).dropna(axis=0)
    filtered_data = filtered_data.reset_index(drop=True)

    print(f"Final dataset of size: {filtered_data.shape}")
    print(filtered_data.columns.to_series().to_string(index=False))
    print(filtered_data.isnull().sum(axis=0))
    print(filtered_data.shape[0])
    return filtered_data


def main():
    read_dataset()


if __name__ == "__main__":
    main()
