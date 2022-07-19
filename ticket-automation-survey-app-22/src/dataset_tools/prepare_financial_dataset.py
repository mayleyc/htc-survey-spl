import re
from pathlib import Path

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from src.utils.torch_train_eval.generic_functions import simplify_text

dataset = Path("data") / "financial.json"


def read_dataset(threshold: int = 30) -> pd.DataFrame:
    """
    Read the dataset from file and return it cleaned of null values/rows

    :param threshold: Minimum numbers of class representatives necessary for a class to be included in the final
                      dataset
    :return: filtered data (as pd.df)
    """
    # Read all data in a DataFrame
    data = pd.read_json(dataset)
    sub_data: pd.DataFrame = data["_source"].apply(pd.Series)
    data = pd.concat([data.drop(columns="_source"), sub_data], axis=1)
    print(data.shape[0])

    # Keep only tickets that have a message description
    data["complaint_what_happened"] = data["complaint_what_happened"].map(lambda x: re.sub(r"XXXX|XX/XX/\d{4}", "", x))
    filtered_data = data[data["complaint_what_happened"].str.len() != 0]
    print(filtered_data.shape[0])
    filtered_data = filtered_data[~filtered_data["complaint_what_happened"].isnull()]
    print(filtered_data.shape[0])
    print(f"Average character number in ticket body: {filtered_data['complaint_what_happened'].str.len().mean():.2f}")

    # Find missing values in product and sub_product
    null_prods = filtered_data["product"].isnull().sum()
    null_subs = filtered_data["sub_product"].isnull().sum()
    print(f"Nan values in 'product': {null_prods}")
    print(f"Nan values in 'sub_product': {null_subs}")

    # Fill missing sub_products fields with product values
    print("Fixing missing values ...")
    filtered_data["sub_product"] = filtered_data["sub_product"].fillna(filtered_data["product"])

    null_prods = filtered_data["product"].isnull().sum()
    null_subs = filtered_data["sub_product"].isnull().sum()
    print(f"Nan values in 'product': {null_prods}")
    print(f"Nan values in 'sub_product': {null_subs}")

    # Clean useless columns
    keep = ["complaint_what_happened", "product", "sub_product"]
    filtered_data = filtered_data.loc[:, keep]
    filtered_data.columns = ["message", "label", "sub_label"]

    # Delete duplicate rows
    filtered_data = filtered_data.drop_duplicates(subset=["message"], keep="first")
    filtered_data = filtered_data.dropna(axis=0)
    # Delete rows having labels that appear less than 'n' times
    # n = 20
    # filtered_data = filtered_data[filtered_data.groupby("label")["label"].transform("count").ge(n)]
    # filtered_data = filtered_data[filtered_data.groupby("sub_label")["sub_label"].transform("count").ge(n)]
    # filtered_data = filtered_data.dropna(axis=0)

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


if __name__ == "__main__":
    read_dataset()
