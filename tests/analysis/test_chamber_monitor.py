import inspect

import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.chamber_monitor import (ALARM_COLUMNS, WINDOW_COLUMNS, alarm_windows, confirm_alarm, monitor,
                                              pass_day_counts)
from fabtrace.analysis.pipeline import DEFECT_PATTERNS
from fabtrace.analysis.spc import detect
from fabtrace.sim.genealogy import generate_genealogy

# 실제 WM-811K 라벨 비율. 원인을 심지 않은 예측을 이 비율로 뽑는다.
REAL_CLASS_COUNTS = {
    "none": 147431, "Edge-Ring": 9680, "Edge-Loc": 5189, "Center": 4294, "Loc": 3593,
    "Scratch": 1193, "Random": 866, "Donut": 555, "Near-full": 149,
}


def natural_predictions(genealogy, rng_seed):
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    classes = list(REAL_CLASS_COUNTS)
    prob = np.array(list(REAL_CLASS_COUNTS.values()), dtype=float)
    prob /= prob.sum()
    return wafers.assign(pred_pattern=np.random.default_rng(rng_seed).choice(classes, len(wafers), p=prob))


def s1_predictions(genealogy, fab, s1, rng_seed):
    """원인 챔버를 12~18일차에 지난 웨이퍼는 0.4, 나머지는 0.05 확률로 Edge-Ring 예측."""
    lo, hi = s1.window(fab.start_date)
    hit = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    rate = np.where(wafers["wafer_id"].isin(hit["wafer_id"]), 0.4, 0.05)
    rng = np.random.default_rng(rng_seed)
    return wafers.assign(pred_pattern=np.where(rng.random(len(wafers)) < rate, "Edge-Ring", "none"))


def test_pass_day_counts_by_chamber_and_day():
    t0 = pd.Timestamp("2026-01-01")
    g = pd.DataFrame({"wafer_id": ["w1", "w2", "w3", "w1"], "step": ["E", "E", "E", "C"],
                      "chamber": ["E/A", "E/A", "E/B", "C/A"],
                      "ts": [t0, t0 + pd.Timedelta(hours=1), t0, t0 + pd.Timedelta(days=1)]})
    preds = pd.DataFrame({"wafer_id": ["w1", "w2", "w3"], "pred_pattern": ["Edge-Ring", "none", "Edge-Ring"]})
    c = pass_day_counts(preds, g, "2026-01-01", ["Edge-Ring"]).set_index(["chamber", "day"])
    assert c.loc[("E/A", 1), "n"] == 2 and c.loc[("E/A", 1), "Edge-Ring"] == 1
    assert c.loc[("E/B", 1), "Edge-Ring"] == 1
    assert c.loc[("C/A", 2), "n"] == 1  # w1이 다음 날 C/A를 지남


def test_tail_spreads_budget_over_all_points(genealogy, fab):
    preds = natural_predictions(genealogy, 0)
    m = monitor(preds, genealogy, fab.start_date, DEFECT_PATTERNS)
    n_chamber_days = genealogy.assign(day=genealogy["ts"].dt.normalize()).groupby(["chamber", "day"]).ngroups
    assert m.n_points == n_chamber_days * len(DEFECT_PATTERNS)
    assert m.tail == pytest.approx(m.budget / m.n_points)
    assert list(m.alarms.columns) == ALARM_COLUMNS


def test_monitor_does_not_take_truth():
    for fn in (pass_day_counts, monitor):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"truth_lots", "truth_wafers", "ground_truth", "true_pattern", "affected"}


def test_false_alarms_at_most_one_per_run(genealogy, fab):
    """원인을 심지 않은 데이터에서 한 번 실행당 알림 구간 수의 평균이 1건 이하다."""
    alarms = [len(monitor(natural_predictions(genealogy, r), genealogy, fab.start_date, DEFECT_PATTERNS).alarms)
              for r in range(20)]
    assert np.mean(alarms) <= 1.0


@pytest.mark.parametrize("genealogy_seed,rng_seed", [(0, 0), (2, 3), (8, 2)])
def test_finds_chamber_excursion_diluted_fab_wide(fab, s1, genealogy_seed, rng_seed):
    """공장 전체 관리도가 놓치는 원인 챔버의 이상을 챔버 감시가 찾는다."""
    g = generate_genealogy(fab, genealogy_seed)
    preds = s1_predictions(g, fab, s1, rng_seed)
    _, windows = detect(preds, g, "Edge-Ring", fab.start_date)
    assert not any(s <= 20 and e >= 12 for s, e in windows)  # 공장 전체로는 이상 구간이 없다

    a = monitor(preds, g, fab.start_date, ["Edge-Ring"]).alarms
    cause = a[(a["chamber"] == s1.chamber) & (a["start_day"] <= 18) & (a["end_day"] >= 12)]
    assert len(cause) >= 1


def test_alarm_windows_converts_to_fab_out_and_merges():
    """E/B를 3일차에 지난 w1, w2는 5, 6일차에, 4일차에 지난 w3는 6일차에 팹 아웃된다."""
    t0 = pd.Timestamp("2026-01-01")
    day = lambda d: t0 + pd.Timedelta(days=d - 1, hours=1)  # noqa: E731
    g = pd.DataFrame({
        "wafer_id": ["w1", "w2", "w3", "w1", "w2", "w3"], "step": ["E"] * 3 + ["C"] * 3,
        "chamber": ["E/B", "E/B", "E/B", "C/A", "C/A", "C/A"], "ts": [day(3), day(3), day(4), day(5), day(6), day(6)]})
    row = {"pattern": "Edge-Ring", "step": "E", "chamber": "E/B", "max_p": 0.5, "center": 0.05}
    alarms = pd.DataFrame([row | {"start_day": 3, "end_day": 3, "alarm_days": 1},
                           row | {"start_day": 4, "end_day": 4, "alarm_days": 1, "max_p": 0.9}])
    w = alarm_windows(alarms, g, "2026-01-01")
    assert list(w.columns) == WINDOW_COLUMNS and len(w) == 1  # 팹 아웃 (5, 6)과 (6, 6)이 겹쳐 하나로 합친다
    r = w.iloc[0]
    assert (r["start_day"], r["end_day"], r["fab_out_start"], r["fab_out_end"]) == (3, 4, 5, 6)
    assert r["alarm_days"] == 2 and r["max_p"] == 0.9


def test_alarm_windows_empty():
    assert alarm_windows(pd.DataFrame(columns=ALARM_COLUMNS), pd.DataFrame(), "2026-01-01").empty


def test_confirm_alarm_accepts_cause_and_rejects_bottleneck(genealogy, fab, s1):
    preds = s1_predictions(genealogy, fab, s1, 0)
    ok, res = confirm_alarm(preds, genealogy, (13, 19), fab.start_date, "Edge-Ring", s1.chamber)
    assert ok and res["B"].iloc[0]["chamber"] == s1.chamber
    ok, _ = confirm_alarm(preds, genealogy, (13, 19), fab.start_date, "Edge-Ring", "CLN-01/A")
    assert not ok  # 병목은 모든 웨이퍼가 지나가므로 비교 대상이 없다
