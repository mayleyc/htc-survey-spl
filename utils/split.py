from operator import itemgetter
from typing import Union, List, Tuple

import numpy as np
from skmultilearn.model_selection import IterativeStratification


def stratified_split(x: Union[np.array, List], y: np.array, splits: Tuple[float, float], order: int = 1,
                     return_indexes=False, shuffle=True) -> Tuple:
    """
    Iteratively stratified train/test split

    :param x: training examples (1D array or Python list)
    :param y: training labels (2D numpy array)
    :param splits: splits ratio (2 floats), must sum to 1
    :param order: see official documentation, for now leave default
    :param return_indexes: whether to return also split indices
    :param shuffle: whether to shuffle

    :return: stratified division into train/test split (as x_train, y_train, x_test, y_test, [idx_train, idx_test])
    """
    if shuffle:
        gen = np.random.default_rng()  # automatically set a random seed
        tmp = zip(x, y)
        tmp = np.array(list(tmp), dtype=object)
        gen.shuffle(tmp)
        x, y = zip(*tmp)
        x = list(x)
        y = np.array(y, dtype=np.int)

    stratifier = IterativeStratification(n_splits=2, order=order, sample_distribution_per_fold=splits)
    split_indexes: Tuple[np.array] = next(stratifier.split(x, y))

    # Fix to order indexes as expected from passed splits
    if not ((splits[0] >= splits[1] and len(split_indexes[0]) >= len(split_indexes[1])) or (
            splits[0] < splits[1] and len(split_indexes[0]) < len(split_indexes[1]))):
        split_indexes = tuple(reversed(split_indexes))

    splits_x = list()
    splits_y = list()
    for indexes in split_indexes:
        if shuffle:
            gen = np.random.default_rng()  # automatically set a random seed
            gen.shuffle(indexes)
        x_s = x[indexes] if isinstance(x, np.ndarray) else itemgetter(*indexes)(x)
        y_s = y[indexes, :]
        splits_x.append(x_s)
        splits_y.append(y_s)

    res = tuple([*splits_x, *splits_y])
    if return_indexes:
        res = *res, *split_indexes
    return res
