"""Split dataset indices into train / val / test, shared by train_network.py and evaluate.py.

Why a shared helper: the two files used to duplicate the same index-slicing code, so the set
`evaluate.py` called "test" coincided exactly with the validation set `train_network.py` used
to select the checkpoint. The reported number was then measured on the set the model was
selected on, not an independent one (both produced exactly 1,763 samples).

Slicing convention::

    [--------------- train ---------------][--- val ---][--- test ---]
    |<------------------ dev ------------------------->|
    |<----------------------- length ------------------------------->|

`test` is taken from the *end* of the list, and `train_frac` applies to the remaining `dev`
portion. With `test_frac=0.0` the function reproduces the previous behaviour exactly
(train = [:train_frac*n], val = the rest), so existing commands give unchanged results.
"""

import numpy as np

SUBSETS = ("train", "val", "test", "dev", "all")


def index_splits(length, train_frac, test_frac=0.0, shuffle=False, seed=123):
    """
    :param length: number of samples in the dataset
    :param train_frac: fraction of the *dev* part (train + val) used for training
    :param test_frac: fraction of the *whole* dataset held out as test, taken from the end
    :param shuffle: shuffle indices before slicing (uses np.random.seed(seed))
    :param seed: seed for the shuffle
    :return: dict {"train": [...], "val": [...], "test": [...]}
    """
    if not 0.0 <= train_frac <= 1.0:
        raise ValueError(f"train_frac must be in [0, 1], got {train_frac}")
    if not 0.0 <= test_frac <= 1.0:
        raise ValueError(f"test_frac must be in [0, 1], got {test_frac}")

    indices = list(range(length))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    n_test = int(np.floor(test_frac * length))
    n_dev = length - n_test
    n_train = int(np.floor(train_frac * n_dev))
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_dev],
        "test": indices[n_dev:],
    }


def select_subset(splits, name):
    """
    :param splits: the result of `index_splits`
    :param name: one of SUBSETS. "dev" = train + val, "all" = everything.
    """
    if name == "dev":
        return splits["train"] + splits["val"]
    if name == "all":
        return splits["train"] + splits["val"] + splits["test"]
    if name not in splits:
        raise ValueError(f"subset must be one of {SUBSETS}, got {name!r}")
    return splits[name]


def describe(splits):
    """A one-line log of how the split came out -- so it can be recovered from the log later."""
    return " · ".join(
        f"{name} {len(idx):,}"
        for name, idx in splits.items()
        if name in ("train", "val", "test")
    )
