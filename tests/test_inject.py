import json
from pathlib import Path

import pandas as pd
import pytest

from fabtrace.sim.config import load_fab_config
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth
from fabtrace.sim.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    fab = load_fab_config(ROOT / "configs" / "fab.yaml")
    sc = load_scenario(ROOT / "configs" / "scenarios" / "s1.yaml", fab)
    gen = generate_genealogy(fab, seed=42)
    # 실제 fab pool과 비슷한 비율의 가짜 pool (CI에서도 돌도록)
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(100_000, 100_000 + len(labels)))
    wafers = build_wafers(gen, sc, fab, pool, seed=42)
    return fab, sc, gen, pool, wafers


def test_affected_wafers_passed_cause_chamber_in_window(setup):
    fab, sc, gen, _, wafers = setup
    lo, hi = sc.window(fab.start_date)
    hits = gen[(gen["chamber"] == sc.chamber) & gen["ts"].between(lo, hi, inclusive="left")]
    assert set(wafers.loc[wafers["affected"], "wafer_id"]) == set(hits["wafer_id"])
    assert 80 < wafers["affected"].sum() < 250


def test_injected_pattern_is_concentrated(setup):
    _, sc, _, _, wafers = setup
    rate = wafers.groupby("affected")["true_pattern"].apply(lambda s: (s == sc.pattern).mean())
    assert rate[True] > rate[False] * 3


def test_maps_unique_and_from_pool(setup):
    _, _, _, pool, wafers = setup
    assert wafers["map_index"].is_unique
    assert set(wafers["map_index"]) <= set(pool.index)
    assigned = pool.loc[wafers["map_index"], "label"].to_numpy()
    assert (assigned == wafers["true_pattern"].to_numpy()).all()


def test_ground_truth_matches_scenario(setup):
    fab, sc, _, _, wafers = setup
    gt = json.loads(json.dumps(ground_truth(sc, fab, wafers, seed=42)))
    assert (gt["chamber"], gt["pattern"]) == ("ETCH-03/B", "Edge-Ring")
    assert (gt["window"]["start_day"], gt["window"]["end_day"]) == (12, 18)
    assert set(gt["affected_lots"]) == set(wafers.loc[wafers["affected"], "lot_id"])


def test_reproducible(setup):
    fab, sc, gen, pool, wafers = setup
    pd.testing.assert_frame_equal(wafers, build_wafers(gen, sc, fab, pool, seed=42))