import inspect

import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.commonality import window_population
from fabtrace.analysis.containment import contain, exposure_window, score
from fabtrace.analysis.exposure import (chamber_daily_counts, chamber_runs, contain_by_chamber,
                                        estimate_exposure, select_runs)

START = "2026-01-01"


def tiny_genealogy():
    """로트 10개, 6시간 간격, 로트마다 2장. 1일차 L0~L3, 2일차 L4~L7, 3일차 L8~L9."""
    t0 = pd.Timestamp(START)
    rows = []
    for i in range(10):
        for s in range(2):
            rows.append({"wafer_id": f"L{i}-W{s}", "lot_id": f"L{i}", "chamber": "E/B" if s == 0 else "E/A",
                         "ts": t0 + pd.Timedelta(hours=6 * i)})
    return pd.DataFrame(rows)


def alarm_chart(days, alarm_days):
    return pd.DataFrame({"alarm": [d in alarm_days for d in days]}, index=pd.Index(days, name="day"))


def test_chamber_daily_counts_uses_only_that_chamber():
    g = tiny_genealogy()
    preds = pd.DataFrame({"wafer_id": g["wafer_id"], "pred_pattern": "none"})
    preds.loc[preds["wafer_id"].isin(["L1-W0", "L5-W0", "L6-W1"]), "pred_pattern"] = "Edge-Ring"
    c = chamber_daily_counts(preds, g, "E/B", "Edge-Ring", START)
    assert c["n"].to_dict() == {1: 4, 2: 4, 3: 2}
    assert c["defects"].to_dict() == {1: 1, 2: 1, 3: 0}  # L6-W1은 E/A를 지났으므로 세지 않는다


def test_chamber_daily_counts_needs_predictions():
    g = tiny_genealogy()
    with pytest.raises(ValueError):
        chamber_daily_counts(g[["wafer_id"]].assign(true_pattern="none"), g, "E/B", "Edge-Ring", START)


def test_runs_join_one_day_gap_only():
    days = list(range(1, 21))
    assert chamber_runs(alarm_chart(days, {5, 6, 8})) == [(5, 8)]
    assert chamber_runs(alarm_chart(days, {5, 6, 9})) == [(5, 6), (9, 9)]


def test_runs_skip_days_without_wafers():
    """챔버를 지난 웨이퍼가 없는 날(7, 8일차)은 간격으로 세지 않는다."""
    days = [d for d in range(1, 21) if d not in (7, 8)]
    assert chamber_runs(alarm_chart(days, {5, 6, 9, 10})) == [(5, 10)]


def test_select_runs():
    runs = [(3, 5), (9, 10), (20, 22)]
    assert select_runs(runs, (4, 4)) == (3, 5)
    assert select_runs(runs, (5, 9)) == (3, 10)
    assert select_runs(runs, (12, 15)) is None


def test_falls_back_to_anchor_without_chamber_alarm():
    g = tiny_genealogy()
    preds = pd.DataFrame({"wafer_id": g["wafer_id"], "pred_pattern": "none"})
    pop = pd.DataFrame({"wafer_id": ["L3-W0", "L5-W0"]})
    ex = estimate_exposure(pop, preds, g, "E/B", "Edge-Ring", START)
    assert ex.method == "anchor" and ex.selected is None
    assert (ex.start, ex.end) == exposure_window(pop, g, "E/B", buffer_hours=0)


def test_does_not_take_truth():
    for fn in (chamber_daily_counts, estimate_exposure, contain_by_chamber):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"truth_lots", "truth_wafers", "ground_truth", "true_pattern", "affected"}


@pytest.mark.parametrize("rng_seed", range(5))
def test_recovers_whole_exposure_from_short_window(genealogy, fab, s1, rng_seed):
    """관리도가 정점 이틀(팹 아웃 15~16일차)만 잡아도, 원인 챔버의 불량률이 오른 기간 전체를 찾는다.

    원인 챔버를 12~18일차에 지난 웨이퍼는 0.4, 나머지는 0.05 확률로 Edge-Ring으로 예측됐다고 둔다.
    """
    lo, hi = s1.window(fab.start_date)
    hit = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    rng = np.random.default_rng(rng_seed)
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    rate = np.where(wafers["wafer_id"].isin(hit["wafer_id"]), 0.4, 0.05)
    wafers["pred_pattern"] = np.where(rng.random(len(wafers)) < rate, "Edge-Ring", "none")
    pop = window_population(wafers, genealogy, (15, 16), fab.start_date, "Edge-Ring")

    c, ex = contain_by_chamber(pop, wafers, genealogy, s1.chamber, "Edge-Ring", fab.start_date, buffer_hours=0)
    assert ex.method == "chamber"
    assert ex.start <= hit["ts"].min() and ex.end >= hit["ts"].max()

    new = score(c, hit["lot_id"].unique(), hit["wafer_id"])
    old = score(contain(pop, genealogy, s1.chamber, buffer_hours=0), hit["lot_id"].unique(), hit["wafer_id"])
    assert new["lot_recall"] == 1.0
    assert new["lot_recall"] > old["lot_recall"]
    assert new["lot_precision"] >= 0.8
