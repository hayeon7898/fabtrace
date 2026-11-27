import pandas as pd
import pytest

from fabtrace.analysis.experiment import (BUFFERS, E4_METHODS, choose_buffer, evaluate_methods, evaluate_seed,
                                          judge_e4, judge_e6, summarize)
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth


def _rows(fab, s1, method):
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    out = []
    for seed in (0, 1):
        gen = generate_genealogy(fab, seed)
        wafers = build_wafers(gen, s1, fab, pool, seed)
        preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
        out.append(evaluate_seed(seed, gen, wafers, ground_truth(s1, fab, wafers, seed, gen), preds, fab.start_date,
                                 method=method))
    return pd.DataFrame(out)


@pytest.fixture(scope="module")
def rows(fab, s1):
    return _rows(fab, s1, "baseline")


@pytest.fixture(scope="module")
def rows_exposure(fab, s1):
    return _rows(fab, s1, "exposure")


def test_rows_have_all_metrics(rows):
    for b in BUFFERS:
        assert {f"lot_recall_{b}h", f"lot_precision_{b}h", f"wafer_recall_{b}h"} <= set(rows.columns)
    assert rows["detected"].all() and rows["cause_found"].all()
    assert (rows["rank_b"] == 1).all()


def test_more_buffer_never_lowers_recall(rows):
    assert (rows["lot_recall_24h"] >= rows["lot_recall_0h"]).all()


def test_summary(rows):
    s = summarize(rows)
    assert s["seeds"] == 2 and s["top1_b"] == "2/2"

def test_exposure_method_reaches_buffer_columns(rows, rows_exposure):
    """여유 시간별 격리 수치도 method를 따른다. 감지와 원인 순위는 같다."""
    assert rows_exposure[["detected", "rank_a", "rank_b"]].equals(rows[["detected", "rank_a", "rank_b"]])
    for b in BUFFERS:
        assert (rows_exposure[f"lot_recall_{b}h"] >= rows[f"lot_recall_{b}h"]).all()
    assert (rows_exposure["lot_recall_0h"] == 1.0).all()


def _review_rows(recalls: dict[int, list[float]]) -> pd.DataFrame:
    n = len(next(iter(recalls.values())))
    return pd.DataFrame({f"lot_recall_{b}h": r for b, r in recalls.items()}
                        | {f"lot_precision_{b}h": [0.5] * n for b in recalls})


def _real(recalls: dict[int, float]) -> dict:
    return {f"lot_recall_{b}h": r for b, r in recalls.items()} | {f"lot_precision_{b}h": 0.5 for b in recalls}


def test_choose_buffer_picks_shortest_passing():
    buffers = (0, 6, 12, 18, 24)
    synth = _review_rows({0: [0.9, 0.95], 6: [0.97, 1.0], 12: [1.0, 1.0], 18: [1.0, 1.0], 24: [1.0, 1.0]})
    real = _real({0: 1.0, 6: 1.0, 12: 1.0, 18: 1.0, 24: 1.0})
    chosen, table = choose_buffer(real, synth, buffers)
    assert chosen == 6  # 6h: 평균 0.985, 최소 0.97
    assert table["passes"].tolist() == [False, True, True, True, True]


def test_choose_buffer_needs_real_recall_one():
    buffers = (0, 6, 12, 18, 24)
    synth = _review_rows({b: [1.0, 1.0] for b in buffers})
    real = _real({0: 0.9, 6: 0.9, 12: 1.0, 18: 1.0, 24: 1.0})
    assert choose_buffer(real, synth, buffers)[0] == 12


def test_choose_buffer_keeps_current_when_nothing_shorter_passes():
    buffers = (0, 6, 12, 18, 24)
    synth = _review_rows({b: [0.85, 1.0] for b in buffers})
    real = _real({b: 1.0 for b in buffers})
    chosen, table = choose_buffer(real, synth, buffers)
    assert chosen == 24 and not table["passes"].any()


@pytest.fixture(scope="module")
def e4_rows(fab, s1):
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    gen = generate_genealogy(fab, 0)
    wafers = build_wafers(gen, s1, fab, pool, 0)
    preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
    gt = ground_truth(s1, fab, wafers, 0, gen)
    return evaluate_methods(0, gen, wafers, gt, preds, fab.start_date), evaluate_seed(
        0, gen, wafers, gt, preds, fab.start_date)


def test_evaluate_methods_adds_e4_metrics(e4_rows):
    rows, base_row = e4_rows
    assert [r["method"] for r in rows] == list(E4_METHODS)
    base = rows[0]
    assert {k: base[k] for k in base_row} == base_row  # baseline은 evaluate_seed와 같다
    for r in rows:
        for b in BUFFERS:
            assert r[f"extra_lots_{b}h"] == r[f"held_lots_{b}h"] - round(r[f"lot_recall_{b}h"] * r["affected_lots"])
        assert r["wrong_causes_all"] >= r["false_causes"]
    assert rows[0]["chamber_alarms"] == 0 and rows[2]["chamber_alarms"] >= rows[2]["alarms_confirmed"]


def _e4(base: list[tuple], new: list[tuple]) -> pd.DataFrame:
    """(detected, lot_recall_24h, extra_lots_24h, wrong_causes_all) 목록으로 E4 행을 만든다."""
    rows = []
    for method, items in (("baseline", base), ("exposure+chamber", new)):
        for seed, (d, r, e, w) in enumerate(items):
            rows.append({"method": method, "seed": seed, "detected": d, "lot_recall_24h": r,
                         "extra_lots_24h": e, "wrong_causes_all": w})
    return pd.DataFrame(rows)


def test_judge_e4_supports_all():
    base = [(True, 0.8, 1, 0)] * 8 + [(False, 0.0, 0, 0)] * 2
    new = [(True, 1.0, 3, 0)] * 9 + [(True, 0.9, 3, 0)]
    v = {j["hypothesis"]: j["verdict"] for j in judge_e4(_e4(base, new))}
    assert v == {1: "지지", 2: "지지", 3: "지지", 4: "지지"}


def test_judge_e4_rejects_on_thresholds():
    base = [(True, 0.8, 0, 0)] * 10
    new = [(True, 0.95, 4, 1)] * 8 + [(True, 0.9, 4, 0)] * 2  # 놓친 시드 10개, 추가 보류 +4, 잘못 지목 8
    v = {j["hypothesis"]: j["verdict"] for j in judge_e4(_e4(base, new))}
    assert v == {1: "판정 불가", 2: "기각", 3: "기각", 4: "기각"}


def test_judge_e4_undecidable_when_one_side_detects_nothing():
    """baseline이 감지한 시드가 없으면 recall 평균을 비교할 수 없으므로 가설 2는 판정 불가다."""
    base = [(False, 0.0, 0, 0)] * 2
    new = [(True, 1.0, 4, 0)] * 2
    v = {j["hypothesis"]: j["verdict"] for j in judge_e4(_e4(base, new))}
    assert v == {1: "지지", 2: "판정 불가", 3: "판정 불가", 4: "지지"}


def test_eval_seeds_are_rejected():
    from fabtrace.analysis.experiment import DEV_SEEDS, EVAL_SEEDS, check_not_eval

    check_not_eval(DEV_SEEDS)
    with pytest.raises(ValueError):
        check_not_eval([200, EVAL_SEEDS[0]])


def _e6(rows: dict[str, list[dict]]) -> pd.DataFrame:
    base = {"c0_found": True, "c0_lot_recall": 1.0, "c0_chamber_fraction": 1.0, "causes_found": 1,
            "n_wrong_causes": 0, "no_cause_correct": None}
    return pd.DataFrame([{"scenario": s, "seed": i} | base | r for s, rs in rows.items() for i, r in enumerate(rs)])


def _all_supporting():
    sweep = {f"S2@{i:g}": [{"c0_found": k < n} for k in range(10)]
             for i, n in ((0.4, 10), (0.3, 8), (0.2, 5), (0.15, 3), (0.1, 1))}
    sweep["S2@0.4"] = [{"c0_found": True, "n_wrong_causes": 2}] * 10
    sweep["S2@0.2"] = [{"c0_found": k < 5, "c0_chamber_fraction": 0.67} for k in range(10)]
    return {"S1": [{}] * 9 + [{"c0_lot_recall": 0.9}],
            "S3": [{"causes_found": 2, "n_wrong_causes": 1}] * 9 + [{"causes_found": 1}],
            "S5": [{"no_cause_correct": True}] * 9 + [{"no_cause_correct": False}]} | sweep


def test_judge_e6_supports_all():
    v = {j["hypothesis"]: j["verdict"] for j in judge_e6(_e6(_all_supporting()))}
    assert v == {h: "지지" for h in range(1, 7)}


def test_judge_e6_rejects_and_handles_missing():
    rows = _all_supporting()
    rows["S2@0.2"] = [{"c0_found": True}] * 9 + [{"c0_found": False}]   # 못 찾은 시드 1개 < 3
    rows["S2@0.1"] = [{"c0_found": True}] * 10                        # 강도가 낮은데 특정률이 오른다
    del rows["S5"]
    v = {j["hypothesis"]: j["verdict"] for j in judge_e6(_e6(rows))}
    assert v[2] == "기각" and v[6] == "기각" and v[4] == "판정 불가" and v[1] == "지지"
