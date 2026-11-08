"""분류기 학습용 데이터 구성. torch에 의존하지 않고 numpy 배열만 다룬다.

- 몫 분리: assign_pools의 train 몫만 학습, val 몫만 검증에 쓴다. fab 몫은 가상 팹 전용이다.
- 인코딩: 맵 값 0(웨이퍼 바깥), 1(정상 다이), 2(불량 다이)를 3채널 원-핫으로 바꾼다.
- 증강: 90도 회전과 좌우 뒤집기 8가지 조합. 보간이 없어 값이 섞이지 않는다.
- 샘플링: 에폭마다 none을 none_cap장만 새로 뽑고 나머지 클래스는 전부 쓴다.
- 클래스 가중치: 에폭당 기대 빈도의 역수 제곱근.

torch 변환은 학습 스크립트에서 torch.from_numpy로 한다.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fabtrace.data.wm811k import CLASSES, assign_pools, load_maps, load_meta

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NONE_ID = CLASS_TO_ID["none"]
N_CLASSES = len(CLASSES)
DEFAULT_NONE_CAP = 10_000


@dataclass(frozen=True)
class WaferData:
    maps: np.ndarray    # (N, 64, 64) uint8
    labels: np.ndarray  # (N,) int64, CLASSES 순서의 클래스 ID
    pools: np.ndarray   # (N,) "train" / "val" / "fab"

    def indices(self, pool: str) -> np.ndarray:
        return np.flatnonzero(self.pools == pool)


def load_wafer_data(data_dir: str | Path = "data/interim") -> WaferData:
    meta = load_meta(data_dir)
    return WaferData(
        maps=load_maps(data_dir),
        labels=meta["label"].map(CLASS_TO_ID).to_numpy(dtype=np.int64),
        pools=assign_pools(meta["label"]).to_numpy(dtype=str),
    )


def encode(maps: np.ndarray) -> np.ndarray:
    """(B, 64, 64) uint8 -> (B, 3, 64, 64) float32 원-핫."""
    return (maps[:, None, :, :] == np.arange(3, dtype=maps.dtype)[None, :, None, None]).astype(np.float32)


def augment(maps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """샘플마다 회전(0, 90, 180, 270도)과 뒤집기 여부를 무작위로 골라 적용한다."""
    out = maps.copy()
    k = rng.integers(0, 4, size=len(maps))
    flip = rng.integers(0, 2, size=len(maps)).astype(bool)
    for rot in range(4):
        for f in (False, True):
            sel = np.flatnonzero((k == rot) & (flip == f))
            if len(sel) == 0:
                continue
            block = np.rot90(maps[sel], rot, axes=(1, 2))
            out[sel] = block[:, :, ::-1] if f else block
    return out


def class_weights(labels: np.ndarray, none_cap: int = DEFAULT_NONE_CAP) -> np.ndarray:
    """에폭당 기대 빈도의 역수 제곱근. 빈도로 가중 평균하면 1이 되도록 맞춘다."""
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    counts[NONE_ID] = min(counts[NONE_ID], none_cap)
    if (counts == 0).any():
        missing = [CLASSES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"학습 데이터에 없는 클래스: {missing}")
    w = 1.0 / np.sqrt(counts)
    w /= (w * counts).sum() / counts.sum()
    return w.astype(np.float32)


class EpochSampler:
    """에폭마다 none을 none_cap장만 새로 뽑는다. (seed, epoch)가 같으면 결과도 같다."""

    def __init__(self, labels: np.ndarray, indices: np.ndarray, none_cap: int = DEFAULT_NONE_CAP, seed: int = 42):
        self.none_idx = indices[labels[indices] == NONE_ID]
        self.other_idx = indices[labels[indices] != NONE_ID]
        self.none_cap = min(none_cap, len(self.none_idx))
        self.seed = seed

    def epoch_indices(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, epoch])
        picked = rng.choice(self.none_idx, size=self.none_cap, replace=False)
        idx = np.concatenate([picked, self.other_idx])
        rng.shuffle(idx)
        return idx

    def __len__(self) -> int:
        return self.none_cap + len(self.other_idx)


def iter_batches(
    data: WaferData, indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(x: (B, 3, 64, 64) float32, y: (B,) int64)를 순서대로 내놓는다. rng가 있으면 증강한다."""
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        maps = data.maps[idx]
        if rng is not None:
            maps = augment(maps, rng)
        yield encode(maps), data.labels[idx]