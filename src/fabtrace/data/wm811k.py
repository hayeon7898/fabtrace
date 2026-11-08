"""변환된 WM-811K 파일 로드와 용도별 분할.

분할은 시드가 고정된 층화 분할이다. 02(분류기 학습)와 01/03(가상 팹)이 같은 함수를 쓰므로,
가상 팹에 배정되는 맵은 분류기 학습에 절대 쓰이지 않는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = (
    "none", "Edge-Ring", "Edge-Loc", "Center", "Loc",
    "Scratch", "Random", "Donut", "Near-full",
)
SPLIT_SEED = 0
POOL_FRACTIONS = (("train", 0.6), ("val", 0.1), ("fab", 0.3))
META_FILE = "wm811k_labeled_64.parquet"
MAPS_FILE = "wm811k_labeled_64.npz"


def load_meta(data_dir: str | Path = "data/interim") -> pd.DataFrame:
    """행 순서(=인덱스)가 npz의 맵 순서와 같은 메타데이터."""
    return pd.read_parquet(Path(data_dir) / META_FILE)


def load_maps(data_dir: str | Path = "data/interim") -> np.ndarray:
    return np.load(Path(data_dir) / MAPS_FILE)["maps"]


def assign_pools(labels: pd.Series, seed: int = SPLIT_SEED) -> pd.Series:
    """클래스별로 섞은 뒤 train 60%, val 10%, fab 30%로 나눈다."""
    rng = np.random.default_rng(seed)
    pools = pd.Series(index=labels.index, dtype=object)
    for cls in sorted(labels.unique()):
        idx = labels.index[labels == cls].to_numpy().copy()
        rng.shuffle(idx)
        start = 0
        for i, (name, frac) in enumerate(POOL_FRACTIONS):
            end = len(idx) if i == len(POOL_FRACTIONS) - 1 else start + round(frac * len(idx))
            pools.loc[idx[start:end]] = name
            start = end
    return pools