"""파이프라인 전체와 정답 누수 방지.

예측은 정답 패턴을 그대로 쓰는 가짜 예측기로 만든다. 분류기 없이 CI에서 끝까지 돈다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.commonality import window_population
from fabtrace.analysis.containment import contain
from fabtrace.analysis.pipeline import run_analysis, score_result
from fabtrace.analysis.report import write_report
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth


@pytest.fixture(scope="module")
def fab_run(genealogy, fab, s1):
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    wafers = build_wafers(genealogy, s1, fab, pool, seed=42)
    gt = ground_truth(s1, fab, wafers, 42, genealogy)
    preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
    return wafers, gt, preds


@pytest.fixture(scope="module")
def result(fab_run, genealogy, fab):
    _, _, preds = fab_run
    return run_analysis(preds, genealogy, fab.start_date)


@pytest.fixture(scope="module")
def result_exposure(fab_run, genealogy, fab):
    _, _, preds = fab_run
    return run_analysis(preds, genealogy, fab.start_date, method="exposure")


@pytest.fixture(scope="module")
def result_chamber(fab_run, genealogy, fab):
    _, _, preds = fab_run
    return run_analysis(preds, genealogy, fab.start_date, method="exposure+chamber")


def _summary(r):
    return (r.windows,
            [(f.pattern, f.window, f.causes, f.ranking_b[["chamber", "p", "q"]].to_dict("list"),
              [c.lots for c in f.containments]) for f in r.findings])


def test_finds_cause_and_contains(result, fab_run):
    wafers, gt, _ = fab_run
    ev = score_result(result, gt, wafers)
    assert ev["detected"] and ev["top1_b"] and ev["cause_found"]
    assert ev["lot_recall"] == 1.0


@pytest.mark.parametrize("method", ["baseline", "exposure", "exposure+chamber"])
def test_no_leak_shuffled_truth_gives_identical_result(fab_run, genealogy, fab, method):
    """정답 컬럼이 섞인 입력을 줘도 분석 결과가 똑같다. 분석은 pred_pattern만 읽는다."""
    wafers, _, preds = fab_run
    with_truth = preds.assign(true_pattern=wafers["true_pattern"], affected=wafers["affected"])
    rng = np.random.default_rng(0)
    poisoned = with_truth.assign(true_pattern=rng.permutation(with_truth["true_pattern"].to_numpy()),
                                 affected=rng.permutation(with_truth["affected"].to_numpy()))
    a = run_analysis(with_truth, genealogy, fab.start_date, method=method)
    b = run_analysis(poisoned, genealogy, fab.start_date, method=method)
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


def test_unknown_method_is_rejected(fab_run, genealogy, fab):
    _, _, preds = fab_run
    with pytest.raises(ValueError):
        run_analysis(preds, genealogy, fab.start_date, method="chamber")


def test_baseline_is_unchanged(result, fab_run, genealogy, fab):
    """기본값은 baseline이고, 격리 범위는 옵션이 생기기 전처럼 contain을 그대로 쓴 결과와 같다."""
    _, _, preds = fab_run
    explicit = run_analysis(preds, genealogy, fab.start_date, method="baseline")
    assert result.method == "baseline" and _summary(result) == _summary(explicit)
    for f in result.findings:
        assert f.exposures == []
        pop = window_population(preds, genealogy, f.window, fab.start_date, f.pattern)
        for c in f.containments:
            direct = contain(pop, genealogy, c.chamber, result.buffer_hours)
            assert (c.start, c.end, c.lots) == (direct.start, direct.end, direct.lots)


def test_exposure_changes_only_containment(result, result_exposure, fab_run):
    """exposure는 감시와 공통성 분석을 그대로 두고, 격리 범위만 넓힌다."""
    wafers, gt, _ = fab_run
    base, exp = score_result(result, gt, wafers), score_result(result_exposure, gt, wafers)
    assert result_exposure.windows == result.windows
    assert [(f.pattern, f.window, f.causes) for f in result_exposure.findings] == \
           [(f.pattern, f.window, f.causes) for f in result.findings]
    for f in result_exposure.findings:
        assert len(f.exposures) == len(f.containments)
    assert exp["lot_recall"] == 1.0 and exp["lot_recall"] >= base["lot_recall"]


def test_report_marks_method_only_when_not_baseline(result, result_exposure, fab_run, tmp_path):
    wafers, gt, _ = fab_run
    meta = {"scenario": "S1", "seed": 42, "model": "fake", "truth_chamber": gt["chamber"],
            "truth_pattern": gt["pattern"], "truth_window": [12, 18]}
    base = write_report(result, score_result(result, gt, wafers), meta, "base", tmp_path, tmp_path)
    exp = write_report(result_exposure, score_result(result_exposure, gt, wafers), meta, "exp", tmp_path, tmp_path)
    base_json = json.loads(base["json"].read_text(encoding="utf-8"))["analysis"]
    exp_json = json.loads(exp["json"].read_text(encoding="utf-8"))["analysis"]
    assert "method" not in base_json and "exposure_estimate" not in json.dumps(base_json)
    assert exp_json["method"] == "exposure"
    assert exp_json["findings"][0]["containment"][0]["exposure_estimate"]["method"] in ("chamber", "anchor")
    assert "격리 방법: `exposure`" in exp["md"].read_text(encoding="utf-8")
    assert "격리 방법" not in base["md"].read_text(encoding="utf-8")


def test_chamber_mode_keeps_fab_wide_findings(result_exposure, result_chamber, fab_run):
    """공장 전체가 이미 찾은 경우, 공장 전체 Finding과 채점은 exposure와 같고 같은 원인을 두 번 내지 않는다."""
    wafers, gt, _ = fab_run
    fab_findings = [f for f in result_chamber.findings if f.source == "fab"]
    assert _summary(result_exposure) == _summary(result_chamber)[:len(_summary(result_exposure))]
    assert len(fab_findings) == len(result_exposure.findings)
    assert score_result(result_chamber, gt, wafers) == score_result(result_exposure, gt, wafers)
    st = result_chamber.chamber_alarms
    assert (st.loc[st["chamber"] == gt["chamber"], "status"] == "covered").all()
    extra = [f for f in result_chamber.findings if f.source == "chamber"]
    assert all(gt["chamber"] not in f.causes for f in extra)


def test_chamber_mode_finds_excursion_missed_fab_wide(fab, s1):
    """공장 전체 관리도가 놓친 원인을 챔버 감시 알림과 공통성 분석으로 찾고 영향 로트를 모두 보류한다."""
    g = generate_genealogy(fab, 0)
    lo, hi = s1.window(fab.start_date)
    hit = g[(g["chamber"] == s1.chamber) & g["ts"].between(lo, hi, inclusive="left")]
    wafers = g[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    rate = np.where(wafers["wafer_id"].isin(hit["wafer_id"]), 0.4, 0.05)
    wafers["pred_pattern"] = np.where(np.random.default_rng(0).random(len(wafers)) < rate, "Edge-Ring", "none")
    wafers["affected"] = wafers["wafer_id"].isin(hit["wafer_id"])
    gt = {"pattern": "Edge-Ring", "chamber": s1.chamber, "window": {"start_day": 12, "end_day": 18},
          "affected_lots": sorted(hit["lot_id"].unique())}

    base = score_result(run_analysis(wafers, g, fab.start_date, method="exposure"), gt, wafers)
    res = run_analysis(wafers, g, fab.start_date, method="exposure+chamber")
    ev = score_result(res, gt, wafers)
    assert not base["detected"]
    assert ev["detected"] and ev["cause_found"] and ev["lot_recall"] == 1.0
    assert all(f.source == "chamber" and f.causes == [s1.chamber] for f in res.findings if f.pattern == "Edge-Ring")


def test_report_shows_chamber_alarms(result_chamber, fab_run, tmp_path):
    wafers, gt, _ = fab_run
    meta = {"scenario": "S1", "seed": 42, "model": "fake", "truth_chamber": gt["chamber"],
            "truth_pattern": gt["pattern"], "truth_window": [12, 18]}
    out = write_report(result_chamber, score_result(result_chamber, gt, wafers), meta, "ch", tmp_path, tmp_path)
    analysis = json.loads(out["json"].read_text(encoding="utf-8"))["analysis"]
    cm = analysis["chamber_monitoring"]
    assert analysis["method"] == "exposure+chamber" and cm["points"] > 0
    assert {a["status"] for a in cm["alarms"]} <= {"confirmed", "rejected", "covered", "short"}
    assert "챔버 감시 알림" in out["md"].read_text(encoding="utf-8")


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

def test_chamber_mode_names_no_cause_without_fault(genealogy, fab):
    """원인을 심지 않은 데이터에서는 챔버 감시가 알려도 원인 챔버를 지목하지 않는다."""
    classes = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Random", "Donut", "Near-full"]
    prob = np.array([147431, 9680, 5189, 4294, 3593, 1193, 866, 555, 149], dtype=float)
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    for seed in range(100, 105):
        preds = wafers.assign(pred_pattern=np.random.default_rng(seed).choice(classes, len(wafers), p=prob / prob.sum()))
        res = run_analysis(preds, genealogy, fab.start_date, method="exposure+chamber")
        assert not [f for f in res.findings if f.source == "chamber"]
        assert "confirmed" not in set(res.chamber_alarms["status"])
