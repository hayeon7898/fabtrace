import json

import pandas as pd
import pytest

from fabtrace.sim.inject import build_wafers, ground_truth


@pytest.fixture(scope="module")
def pool():
    """실제 fab 몫과 비슷한 비율의 작은 가짜 pool. 인덱스는 npz 맵 위치 역할."""
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    return pd.DataFrame({"label": labels}, index=range(100_000, 100_000 + len(labels)))


@pytest.fixture(scope="module")
def wafers(genealogy, s1, fab, pool):
    return build_wafers(genealogy, s1, fab, pool, seed=42)


def test_affected_wafers_passed_cause_chamber_in_window(fab, s1, genealogy, wafers):
    lo, hi = s1.window(fab.start_date)
    hits = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    assert set(wafers.loc[wafers["affected"], "wafer_id"]) == set(hits["wafer_id"])
    assert 80 < wafers["affected"].sum() < 250


def test_injected_pattern_is_concentrated(s1, wafers):
    rate = wafers.groupby("affected")["true_pattern"].apply(lambda s: (s == s1.pattern).mean())
    assert rate[True] > rate[False] * 3


def test_maps_unique_and_from_pool(pool, wafers):
    assert wafers["map_index"].is_unique
    assert set(wafers["map_index"]) <= set(pool.index)
    assigned = pool.loc[wafers["map_index"], "label"].to_numpy()
    assert (assigned == wafers["true_pattern"].to_numpy()).all()


def test_ground_truth_matches_scenario(fab, s1, wafers):
    gt = json.loads(json.dumps(ground_truth(s1, fab, wafers, seed=42)))
    assert (gt["chamber"], gt["pattern"]) == ("ETCH-03/B", "Edge-Ring")
    assert (gt["window"]["start_day"], gt["window"]["end_day"]) == (12, 18)
    assert set(gt["affected_lots"]) == set(wafers.loc[wafers["affected"], "lot_id"])


def test_reproducible(genealogy, s1, fab, pool, wafers):
    pd.testing.assert_frame_equal(wafers, build_wafers(genealogy, s1, fab, pool, seed=42))