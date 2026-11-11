import inspect

import numpy as np
import pandas as pd

from fabtrace.analysis.commonality import window_population
from fabtrace.analysis.containment import contain, exposure_window, score, trace


def tiny_genealogy():
    t0 = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(10):  # 로트 10개, 6시간 간격, 로트마다 2장
        for s in range(2):
            rows.append({"wafer_id": f"L{i}-W{s}", "lot_id": f"L{i}", "chamber": "E/B" if s == 0 else "E/A",
                         "ts": t0 + pd.Timedelta(hours=6 * i)})
    return pd.DataFrame(rows)


def test_trace_finds_all_in_range():
    g = tiny_genealogy()
    c = trace(g, "E/B", pd.Timestamp("2026-01-01 12:00"), pd.Timestamp("2026-01-02 06:00"))
    assert c.lots == ["L2", "L3", "L4", "L5"]
    assert set(c.wafers["wafer_id"]) == {"L2-W0", "L3-W0", "L4-W0", "L5-W0"}


def test_exposure_window_adds_buffer():
    g = tiny_genealogy()
    pop = pd.DataFrame({"wafer_id": ["L3-W0", "L5-W0", "L4-W1"]})
    start, end = exposure_window(pop, g, "E/B", buffer_hours=6)
    assert start == pd.Timestamp("2026-01-01 12:00") and end == pd.Timestamp("2026-01-02 12:00")


def test_tracing_does_not_take_truth():
    for fn in (exposure_window, trace, contain):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"truth_lots", "truth_wafers", "ground_truth", "true_pattern"}


def test_score():
    g = tiny_genealogy()
    c = trace(g, "E/B", pd.Timestamp("2026-01-01 12:00"), pd.Timestamp("2026-01-02 06:00"))
    s = score(c, truth_lots=["L3", "L4", "L5", "L6"], truth_wafers=["L3-W0", "L4-W0", "L5-W0", "L6-W0"])
    assert s["lot_precision"] == 0.75 and s["lot_recall"] == 0.75
    assert s["missed_lots"] == ["L6"] and s["extra_lots"] == ["L2"]


def test_buffer_recovers_all_affected_lots(genealogy, fab, s1):
    """관리도가 하루 늦게 알아챈 상황(팹 아웃 13~18일차)에서도 여유 하루면 영향 로트를 전부 찾는다."""
    lo, hi = s1.window(fab.start_date)
    hit = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    rng = np.random.default_rng(0)
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    wafers["pred_pattern"] = np.where(rng.random(len(wafers)) < 0.05, "Edge-Ring", "none")
    pop = window_population(wafers, genealogy, (13, 18), fab.start_date, "Edge-Ring")

    tight = score(contain(pop, genealogy, s1.chamber, buffer_hours=0), hit["lot_id"].unique(), hit["wafer_id"])
    loose = score(contain(pop, genealogy, s1.chamber, buffer_hours=24), hit["lot_id"].unique(), hit["wafer_id"])
    assert loose["lot_recall"] == 1.0
    assert loose["lot_recall"] >= tight["lot_recall"]
    assert loose["held_lots"] >= tight["held_lots"]