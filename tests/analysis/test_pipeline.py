"""파이프라인 전체와 정답 누수 방지.

예측은 정답 패턴을 그대로 쓰는 가짜 예측기로 만든다. 분류기 없이 CI에서 끝까지 돈다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.pipeline import run_analysis, score_result
from fabtrace.analysis.report import write_report
from fabtrace.sim.inject import build_wafers, ground_truth


@pytest.fixture(scope="module")
def fab_run(genealogy, fab, s1):
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    wafers = build_wafers(genealogy, s1, fab, pool, seed=42)
    gt = ground_truth(s1, fab, wafers, seed=42)
    preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
    return wafers, gt, preds


@pytest.fixture(scope="module")
def result(fab_run, genealogy, fab):
    _, _, preds = fab_run
    return run_analysis(preds, genealogy, fab.start_date)


def _summary(r):
    return (r.windows,
            [(f.pattern, f.window, f.causes, f.ranking_b[["chamber", "p", "q"]].to_dict("list"),
              [c.lots for c in f.containments]) for f in r.findings])


def test_finds_cause_and_contains(result, fab_run):
    wafers, gt, _ = fab_run
    ev = score_result(result, gt, wafers)
    assert ev["detected"] and ev["top1_b"] and ev["cause_found"]
    assert ev["lot_recall"] == 1.0


def test_no_leak_shuffled_truth_gives_identical_result(fab_run, genealogy, fab):
    """정답 컬럼이 섞인 입력을 줘도 분석 결과가 똑같다. 분석은 pred_pattern만 읽는다."""
    wafers, _, preds = fab_run
    with_truth = preds.assign(true_pattern=wafers["true_pattern"], affected=wafers["affected"])
    rng = np.random.default_rng(0)
    poisoned = with_truth.assign(true_pattern=rng.permutation(with_truth["true_pattern"].to_numpy()),
                                 affected=rng.permutation(with_truth["affected"].to_numpy()))
    a = run_analysis(with_truth, genealogy, fab.start_date)
    b = run_analysis(poisoned, genealogy, fab.start_date)
    assert _summary(a) == _summary(b)


def test_no_cause_gives_not_confirmed(fab_run, genealogy, fab):
    """결함 없이 평상시 분포만 있으면 원인 후보가 거의 나오지 않는다."""
    wafers, _, _ = fab_run
    rng = np.random.default_rng(1)
    preds = wafers[["wafer_id"]].assign(pred_pattern=rng.permutation(wafers["true_pattern"].to_numpy()))
    r = run_analysis(preds, genealogy, fab.start_date)
    assert sum(len(f.causes) for f in r.findings) <= 1


def test_report_outputs(result, fab_run, tmp_path):
    wafers, gt, _ = fab_run
    ev = score_result(result, gt, wafers)
    meta = {"scenario": "S1", "seed": 42, "model": "fake", "truth_chamber": gt["chamber"],
            "truth_pattern": gt["pattern"], "truth_window": [12, 18]}
    paths = write_report(result, ev, meta, "s1_rca", tmp_path / "reports", tmp_path / "figures")
    md = paths["md"].read_text(encoding="utf-8")
    assert "ETCH-03/B" in md and "채점 (여기만 정답지를 사용한다)" in md
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["analysis"]["findings"][0]["causes"] == ["ETCH-03/B"]
    assert "true_pattern" not in json.dumps(payload["analysis"])
    assert all(Path(p).exists() for p in paths.values())


@pytest.mark.needs_data
def test_real_run(real_data_dir, s1_path, tmp_path):
    from fabtrace.analysis.pipeline import simulate_and_predict
    from fabtrace.data.dataset import load_wafer_data

    model = Path("models/cnn_v1.pt")
    if not model.exists():
        pytest.skip("models/cnn_v1.pt 없음")
    fab_cfg, gen, wafers, gt, preds = simulate_and_predict(
        s1_path, s1_path.parents[1] / "fab.yaml", 42, load_wafer_data(real_data_dir), model)
    ev = score_result(run_analysis(preds, gen, fab_cfg.start_date), gt, wafers)
    assert ev["top1_b"]