from pathlib import Path
from typing import List, Callable, Union, Any

import yaml


def load_yaml(path: Union[str, Path]) -> Any:
    """
    Load YAML as python dict

    :param path: path to YAML file
    :return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def to_str(bytes_or_str):
    """
    Helper function to take a bytes or str instance and always return a str. [credit: Effective python]
    """
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value  # Instance of str


def to_bytes(bytes_or_str):
    """
    Helper function to take a bytes or str instance and always return a bytes. [credit: Effective python]
    """
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value  # Instance of bytes


def apply_filters(s: str, filters: List[Callable[[str], str]]) -> str:
    """
    Helper function to apply filters.
    Parameters
    ----------
    s : string being filtered
    filters : list of callable filters to apply to the string

    Returns
    -------
    filtered string
    """
    for f in filters:
        s = f(s).strip()
    return s
