"""실제 데이터로 학습 데이터 구성을 점검한다. 몫별 개수, 1에폭 클래스 분포, 가중치, 배치 생성 시간.

사용법: uv run python scripts/check_training_data.py
"""

import time

import numpy as np
import pandas as pd

from fabtrace.data.dataset import EpochSampler, class_weights, iter_batches, load_wafer_data
from fabtrace.data.wm811k import CLASSES


def main() -> None:
    data = load_wafer_data()
    print({p: len(data.indices(p)) for p in ("train", "val", "fab")})

    train = data.indices("train")
    sampler = EpochSampler(data.labels, train, seed=42)
    idx = sampler.epoch_indices(0)
    weights = class_weights(data.labels[train])

    table = pd.DataFrame({
        "train 몫": np.bincount(data.labels[train], minlength=len(CLASSES)),
        "1에폭": np.bincount(data.labels[idx], minlength=len(CLASSES)),
        "가중치": weights.round(2),
    }, index=CLASSES)
    print(table.to_string())

    start = time.perf_counter()
    n = sum(len(y) for _, y in iter_batches(data, idx, batch_size=256, rng=np.random.default_rng(0)))
    print(f"1에폭 {n:,}장 배치 생성(증강 포함): {time.perf_counter() - start:.1f}초")


if __name__ == "__main__":
    main()