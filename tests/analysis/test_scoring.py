import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.containment import Containment
from fabtrace.analysis.pipeline import Finding, RCAResult, run_analysis, score_result
from fabtrace.analysis.scoring import score_general, score_row
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth

T0 = pd.Timestamp("2026-01-01")
ALL = ["ETCH-02/A", "ETCH-02/B", "ETCH-02/C", "ETCH-03/B", "CMP-01/A", "OX-01/A"]


def _cause(location, chambers, pattern="Edge-Ring", unit="chamber", lots=("L1", "L2"), days=(12, 18)):
    return {"unit": unit, "location": location, "chambers": list(chambers), "pattern": pattern,
            "window": {"start_day": days[0], "end_day": days[1],
                       "start_ts": (T0 + pd.Timedelta(days=days[0] - 1)).isoformat(),
                       "end_ts": (T0 + pd.Timedelta(days=days[1])).isoformat()},
            "affected_lots": list(lots)}


def _gt(*causes):
    return {"causes": list(causes), "affected_lots": sorted({lot for c in causes for lot in c["affected_lots"]})}


def _hold(chamber, lots):
    return Containment(chamber, T0, T0, 24, list(lots), pd.DataFrame({"wafer_id": [f"{x}-W0" for x in lots]}))


def _finding(causes, holds, pattern="Edge-Ring", window=(13, 18), order=ALL):
    ranking = pd.DataFrame({"chamber": order, "rank": range(1, len(order) + 1)})
    return Finding(pattern, window, 100, 10, ranking, ranking, list(causes), list(holds))


def _result(*findings):
    return RCAResult(start_date="2026-01-01", buffer_hours=24, findings=list(findings))


GEN = pd.DataFrame({"wafer_id": ["L1-W0", "L2-W0", "L3-W0"], "chamber": ["ETCH-03/B", "ETCH-02/A", "CMP-01/A"],
                    "ts": [T0 + pd.Timedelta(days=13)] * 3})
WAFERS = pd.DataFrame({"wafer_id": ["L1-W0", "L2-W0", "L3-W0"], "lot_id": ["L1", "L2", "L3"]})


def test_chamber_cause_found():
    gt = _gt(_cause("ETCH-03/B", ["ETCH-03/B"], lots=["L1"]))
    s = score_general(_result(_finding(["ETCH-03/B"], [_hold("ETCH-03/B", ["L1", "L9"])],
                                       order=["ETCH-03/B"] + ALL[:3])), gt, WAFERS, GEN)
    (c,) = s["causes"]
    assert c["found"] and c["detected"] and c["rank_b"] == 1
    assert (c["lot_recall"], c["lot_precision"], c["extra_lots_matched"]) == (1.0, 0.5, 1)
    assert s["cause_recall"] == 1.0 and s["cause_precision"] == 1.0 and s["wrong_causes"] == []


def test_tool_cause_needs_every_chamber_or_tool_name():
    tool = _cause("ETCH-02", ["ETCH-02/A", "ETCH-02/B", "ETCH-02/C"], unit="tool")
    partial = score_general(_result(_finding(["ETCH-02/B"], [])), _gt(tool), WAFERS, GEN)
    assert not partial["causes"][0]["found"]
    assert partial["causes"][0]["chamber_fraction"] == pytest.approx(1 / 3)
    assert partial["cause_precision"] == 1.0  # 일부라도 지목한 챔버 자체는 맞다
    by_name = score_general(_result(_finding(["ETCH-02"], [])), _gt(tool), WAFERS, GEN)
    assert by_name["causes"][0]["found"] and by_name["named_causes"] == 3


def test_two_causes_one_found_and_wrong_causes():
    a = _cause("ETCH-03/B", ["ETCH-03/B"], lots=["L1"])
    b = _cause("CMP-01/A", ["CMP-01/A"], lots=["L3"], days=(15, 22))
    r = _result(_finding(["ETCH-03/B", "OX-01/A"], [_hold("ETCH-03/B", ["L1"]), _hold("OX-01/A", ["L5", "L6"])]),
                _finding(["CMP-01/A"], [], pattern="Center"))  # 패턴이 달라 원인 b를 찾은 것이 아니다
    s = score_general(r, _gt(a, b), WAFERS, GEN)
    assert [c["found"] for c in s["causes"]] == [True, False]
    assert s["cause_recall"] == 0.5
    assert s["wrong_causes"] == [("Center", "CMP-01/A"), ("Edge-Ring", "OX-01/A")]
    assert s["extra_lots_total"] == 2 and s["causes"][0]["extra_lots_matched"] == 0
    assert s["lot_recall_total"] == 0.5


def test_no_cause_scenario():
    assert score_general(_result(), _gt(), WAFERS, GEN)["no_cause_correct"] is True
    s = score_general(_result(_finding(["OX-01/A"], [_hold("OX-01/A", ["L5"])])), _gt(), WAFERS, GEN)
    assert s["no_cause_correct"] is False and s["cause_recall"] is None and s["extra_lots_total"] == 1


def test_score_row_flattens():
    gt = _gt(_cause("ETCH-03/B", ["ETCH-03/B"], lots=["L1"]))
    row = score_row(_result(_finding(["ETCH-03/B", "OX-01/A"], [])), gt, WAFERS, GEN)
    assert row["c0_found"] and row["n_wrong_causes"] == 1 and row["wrong_causes"] == "Edge-Ring:OX-01/A"


@pytest.mark.parametrize("method", ["baseline", "exposure+chamber"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_score_result_on_s1(fab, s1, method, seed):
    """원인이 챔버 하나인 S1에서는 원인 특정과 격리 값이 score_result와 같다."""
    counts = {"none": 11000, "Edge-Ring": 1200, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    gen = generate_genealogy(fab, seed)
    wafers = build_wafers(gen, s1, fab, pool, seed)
    rng = np.random.default_rng(seed)
    noisy = np.where(rng.random(len(wafers)) < 0.03, "Edge-Loc", wafers["true_pattern"])
    preds = wafers[["wafer_id"]].assign(pred_pattern=noisy)
    gt = ground_truth(s1, fab, wafers, seed, gen)
    r = run_analysis(preds, gen, fab.start_date, method=method)
    old, new = score_result(r, gt, wafers), score_general(r, gt, wafers, gen)
    (c,) = new["causes"]
    assert c["detected"] == old["detected"]
    assert c["found_first"] == old["cause_found"]
    if old["detected"]:
        assert c["rank_b"] == old["rank_b"] and c["window"] == old["window"]
    for k in ("lot_recall", "lot_precision", "wafer_recall", "held_lots"):
        if k in old:
            assert c[f"{k}_first"] == pytest.approx(old[k]), k
    assert c["lot_recall"] >= c["lot_recall_first"]  # 구간을 모두 합치면 recall이 줄지 않는다


def test_union_over_windows_counts_every_hold():
    """한 원인에 이상 구간이 둘이면 두 구간의 보류를 모두 센다. 첫 구간 기준 값은 첫 구간만 본다."""
    gt = _gt(_cause("ETCH-03/B", ["ETCH-03/B"], lots=["L1", "L2"]))
    order = ["ETCH-03/B"] + ALL[:3]
    r = _result(_finding(["ETCH-03/B"], [_hold("ETCH-03/B", ["L1"])], window=(13, 13), order=order),
                _finding(["ETCH-03/B"], [_hold("ETCH-03/B", ["L2"])], window=(16, 17), order=order))
    (c,) = score_general(r, gt, WAFERS, GEN)["causes"]
    assert c["lot_recall"] == 1.0 and c["lot_recall_first"] == 0.5
