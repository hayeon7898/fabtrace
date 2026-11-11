import numpy as np
import pandas as pd
import pytest
import torch

from fabtrace.data.dataset import WaferData
from fabtrace.data.wm811k import CLASSES, assign_pools
from fabtrace.models.evaluate import evaluate, predict_wafers, run_evaluation
from fabtrace.models.train import TrainConfig, fit


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


@pytest.fixture(scope="module")
def models(tiny, tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    paths = []
    for balance, name in [(True, "a"), (False, "b")]:
        ckpt = fit(tiny, TrainConfig(balance=balance, max_epochs=1, batch_size=64, none_cap=100), log=lambda *_: None)
        torch.save(ckpt, d / f"{name}.pt")
        paths.append(d / f"{name}.pt")
    return paths


@pytest.fixture(scope="module")
def wafers(tiny):
    fab = tiny.indices("fab")[:50]
    return pd.DataFrame({"wafer_id": [f"W{i:03d}" for i in range(len(fab))], "map_index": fab})


def test_evaluate_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    ev = evaluate("x", {}, y_true, y_pred)
    assert ev.accuracy == 0.75
    assert ev.per_class.loc["none", "recall"] == 0.5
    assert ev.per_class.loc["Edge-Ring", "recall"] == 1.0
    assert ev.confusion.shape == (9, 9)


def test_predict_wafers_covers_all(tiny, models, wafers):
    preds = predict_wafers(models[0], tiny, wafers)
    assert list(preds["wafer_id"]) == list(wafers["wafer_id"])
    probs = preds[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert (preds["confidence"] == probs.max(axis=1)).all()
    assert set(preds["pred_pattern"]) <= set(CLASSES)


def test_run_evaluation_writes_outputs(tiny, models, wafers, tmp_path):
    run_evaluation(models, tiny, wafers, models[0], tmp_path / "results", tmp_path / "processed")
    report = (tmp_path / "results" / "classifier_report.md").read_text(encoding="utf-8")
    assert "fab macro-F1" in report and "Edge-Ring recall" in report
    assert "정확도는 메인 지표로 쓰지 않는다" in report
    assert (tmp_path / "processed" / "predictions.parquet").exists()
    assert (tmp_path / "results" / "figures" / "classifier_learning_curves.png").exists()
    assert len(list((tmp_path / "results" / "figures").glob("confusion_*.png"))) == 2