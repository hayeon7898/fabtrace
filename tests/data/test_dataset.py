import numpy as np
import pandas as pd
import pytest

from fabtrace.data.dataset import (
    N_CLASSES, NONE_ID, EpochSampler, WaferData, augment, class_weights, encode, iter_batches,
)
from fabtrace.data.wm811k import CLASSES, assign_pools


@pytest.fixture(scope="module")
def data():
    """클래스마다 개수가 다른 작은 가짜 데이터. none이 압도적으로 많다."""
    rng = np.random.default_rng(0)
    counts = {c: 30 for c in CLASSES}
    counts["none"] = 3000
    names = [c for c, n in counts.items() for _ in range(n)]
    labels = pd.Series(names)
    return WaferData(
        maps=rng.integers(0, 3, size=(len(names), 64, 64), dtype=np.uint8),
        labels=labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64),
        pools=assign_pools(labels).to_numpy(dtype=str),
    )


def test_train_indices_are_only_train_pool(data):
    train, val, fab = data.indices("train"), data.indices("val"), data.indices("fab")
    assert (data.pools[train] == "train").all()
    assert not set(train) & (set(val) | set(fab))
    assert len(train) + len(val) + len(fab) == len(data.labels)


def test_sampler_uses_only_given_indices(data):
    train = data.indices("train")
    idx = EpochSampler(data.labels, train, none_cap=500).epoch_indices(0)
    assert set(idx) <= set(train)


def test_augment_keeps_values_and_shape(data):
    maps = data.maps[:64]
    out = augment(maps, np.random.default_rng(1))
    assert out.shape == maps.shape and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 1, 2}
    # 회전과 뒤집기는 다이를 옮기기만 하므로 샘플마다 값의 개수가 같다
    assert (np.sort(out.reshape(64, -1), axis=1) == np.sort(maps.reshape(64, -1), axis=1)).all()


def test_augment_is_one_of_eight_symmetries(data):
    m = data.maps[:1]
    out = augment(m, np.random.default_rng(3))[0]
    candidates = [np.rot90(m[0], k) for k in range(4)]
    candidates += [c[:, ::-1] for c in candidates]
    assert any((out == c).all() for c in candidates)


def test_encode_is_one_hot(data):
    x = encode(data.maps[:8])
    assert x.shape == (8, 3, 64, 64) and x.dtype == np.float32
    assert (x.sum(axis=1) == 1).all()
    assert (x.argmax(axis=1) == data.maps[:8]).all()


def test_class_weights(data):
    w = class_weights(data.labels[data.indices("train")], none_cap=500)
    assert w.shape == (N_CLASSES,) and (w > 0).all()
    assert w[NONE_ID] == w.min()


def test_class_weights_missing_class_raises():
    with pytest.raises(ValueError):
        class_weights(np.zeros(10, dtype=np.int64))


def test_sampler_caps_none_and_keeps_others(data):
    train = data.indices("train")
    sampler = EpochSampler(data.labels, train, none_cap=500, seed=42)
    idx = sampler.epoch_indices(0)
    assert (data.labels[idx] == NONE_ID).sum() == 500
    assert set(sampler.other_idx) <= set(idx)
    assert len(idx) == len(sampler)


def test_sampler_reproducible_and_varies_by_epoch(data):
    sampler = EpochSampler(data.labels, data.indices("train"), none_cap=500, seed=42)
    assert (sampler.epoch_indices(0) == sampler.epoch_indices(0)).all()
    assert not (sampler.epoch_indices(0) == sampler.epoch_indices(1)).all()


def test_iter_batches_shapes(data):
    idx = data.indices("val")
    batches = list(iter_batches(data, idx, batch_size=100, rng=np.random.default_rng(0)))
    assert sum(len(y) for _, y in batches) == len(idx)
    x, y = batches[0]
    assert x.shape[1:] == (3, 64, 64) and y.dtype == np.int64