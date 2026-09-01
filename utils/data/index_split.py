"""Chia index dataset thành train / val / test, dùng chung giữa train_network.py và evaluate.py.

Lý do phải dùng chung một hàm: trước đây hai file lặp lại cùng một đoạn code cắt index, nên
tập mà `evaluate.py` gọi là "test" trùng khít tập validation mà `train_network.py` dùng để
chọn checkpoint. Số báo cáo khi đó là số trên tập đã chọn model, không phải tập độc lập --
xem `results/260831_0912_ga-pp-200k-eval-best67/` (cả hai đều ra đúng 1.763 mẫu).

Quy ước cắt::

    [--------------- train ---------------][--- val ---][--- test ---]
    |<------------------ dev ------------------------->|
    |<----------------------- length ------------------------------->|

`test` cắt ở *cuối* danh sách, `train_frac` áp lên phần `dev` còn lại. Với `test_frac=0.0`
hàm cho ra đúng hành vi cũ (train = [:train_frac*n], val = phần còn lại), nên mọi lệnh chạy
sẵn có giữ nguyên kết quả.
"""

import numpy as np

SUBSETS = ("train", "val", "test", "dev", "all")


def index_splits(length, train_frac, test_frac=0.0, shuffle=False, seed=123):
    """
    :param length: số sample của dataset
    :param train_frac: tỉ lệ phần *dev* (train + val) dùng để train
    :param test_frac: tỉ lệ *toàn bộ* dataset giữ làm test, cắt ở cuối danh sách
    :param shuffle: xáo index trước khi cắt (dùng np.random.seed(seed))
    :param seed: seed cho phép xáo
    :return: dict {"train": [...], "val": [...], "test": [...]}
    """
    if not 0.0 <= train_frac <= 1.0:
        raise ValueError(f"train_frac phải nằm trong [0, 1], nhận {train_frac}")
    if not 0.0 <= test_frac <= 1.0:
        raise ValueError(f"test_frac phải nằm trong [0, 1], nhận {test_frac}")

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
    :param splits: kết quả của `index_splits`
    :param name: một trong SUBSETS. "dev" = train + val, "all" = toàn bộ.
    """
    if name == "dev":
        return splits["train"] + splits["val"]
    if name == "all":
        return splits["train"] + splits["val"] + splits["test"]
    if name not in splits:
        raise ValueError(f"subset phải là một trong {SUBSETS}, nhận {name!r}")
    return splits[name]


def describe(splits):
    """Một dòng log cho biết đã cắt ra bao nhiêu -- để đọc lại được từ log về sau."""
    return " · ".join(
        f"{name} {len(idx):,}"
        for name, idx in splits.items()
        if name in ("train", "val", "test")
    )
