"""작은 가짜 데이터로 학습 루프가 끝까지 도는지 확인한다. 실제 학습은 오래 걸려 CI에서 하지 않는다."""

import numpy as np
import pandas as pd
import pytest
import torch

from fabtrace.data.dataset import WaferData
from fabtrace.data.wm811k import CLASSES, assign_pools
from fabtrace.models.train import TrainConfig, fit, load_model, predict_proba


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    counts = {c: 20 for c in CLASSES}
    counts["none"] = 300
    names = [c for c, n in counts.items() for _ in range(n)]
    labels = pd.Series(names)
    return WaferData(
        maps=rng.integers(0, 3, size=(len(names), 64, 64), dtype=np.uint8),
        labels=labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64),
        pools=assign_pools(labels).to_numpy(dtype=str),
    )


CFG = dict(max_epochs=2, patience=5, batch_size=64, none_cap=100)


@pytest.mark.parametrize("balance", [True, False])
def test_fit_runs(tiny, balance):
    ckpt = fit(tiny, TrainConfig(balance=balance, **CFG), log=lambda *_: None)
    assert len(ckpt["history"]) == 2
    assert 0.0 <= ckpt["best_val_macro_f1"] <= 1.0
    assert ckpt["classes"] == list(CLASSES)
    assert ckpt["config"]["balance"] is balance


def test_fit_is_reproducible(tiny):
    a = fit(tiny, TrainConfig(**CFG), log=lambda *_: None)
    b = fit(tiny, TrainConfig(**CFG), log=lambda *_: None)
    assert [r["train_loss"] for r in a["history"]] == [r["train_loss"] for r in b["history"]]


def test_save_and_load(tiny, tmp_path):
    ckpt = fit(tiny, TrainConfig(**CFG), log=lambda *_: None)
    torch.save(ckpt, tmp_path / "m.pt")
    model, loaded = load_model(tmp_path / "m.pt")
    probs = predict_proba(model, tiny, tiny.indices("val"))
    assert probs.shape == (len(tiny.indices("val")), len(CLASSES))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert loaded["best_epoch"] >= 1