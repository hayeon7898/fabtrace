import hashlib
import json

import pandas as pd
import pytest

from fabtrace.sim.inject import build_wafers, cause_passes, ground_truth
from fabtrace.sim.scenario import load_scenario


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


def test_ground_truth_matches_scenario(fab, s1, wafers, genealogy):
    gt = json.loads(json.dumps(ground_truth(s1, fab, wafers, 42, genealogy)))
    assert (gt["chamber"], gt["pattern"]) == ("ETCH-03/B", "Edge-Ring")
    assert (gt["window"]["start_day"], gt["window"]["end_day"]) == (12, 18)
    assert set(gt["affected_lots"]) == set(wafers.loc[wafers["affected"], "lot_id"])
    (c,) = gt["causes"]
    assert (c["unit"], c["location"], c["chambers"]) == ("chamber", "ETCH-03/B", ["ETCH-03/B"])
    assert c["affected_lots"] == gt["affected_lots"]


def test_reproducible(genealogy, s1, fab, pool, wafers):
    pd.testing.assert_frame_equal(wafers, build_wafers(genealogy, s1, fab, pool, seed=42))


# 원인 목록 이전(main 4b906dc)의 코드로 만든 S1 seed 42 wafers의 해시. 주입 코드를 바꿔도 S1은 그대로여야 한다.
S1_WAFERS_SHA256 = "201b61696701ee91ca668575343208e55089c93237abeb4318994fa15b77c4ba"


def test_s1_wafers_unchanged(wafers):
    assert hashlib.sha256(wafers.to_csv(index=False).encode()).hexdigest() == S1_WAFERS_SHA256


@pytest.fixture(scope="module")
def big_pool():
    """강한 주입에도 맵이 모자라지 않게 불량 맵을 넉넉히 둔 pool."""
    counts = {"none": 12000, "Edge-Ring": 3000, "Center": 2500, "Edge-Loc": 600, "Loc": 500}
    labels = [c for c, n in counts.items() for _ in range(n)]
    return pd.DataFrame({"label": labels}, index=range(len(labels)))


def _scenario(tmp_path, fab, causes: str):
    path = tmp_path / "sc.yaml"
    path.write_text("id: T\ncauses:" + causes, encoding="utf-8")
    return load_scenario(path, fab)


def _cause(unit, step, location, start, end, pattern, intensity):
    return (f"\n  - {{unit: {unit}, step: {step}, location: {location}, "
            f"window: {{start_day: {start}, end_day: {end}}}, pattern: {pattern}, intensity: {intensity}}}")


def test_tool_cause_affects_all_its_chambers(tmp_path, fab, genealogy, pool):
    sc = _scenario(tmp_path, fab, _cause("tool", "ETCH", "ETCH-02", 12, 18, "Edge-Ring", 0.4))
    w = build_wafers(genealogy, sc, fab, pool, seed=1)
    lo, hi = sc.causes[0].window(fab.start_date)
    hits = genealogy[genealogy["chamber"].str.startswith("ETCH-02/") & genealogy["ts"].between(lo, hi, inclusive="left")]
    assert set(w.loc[w["affected"], "wafer_id"]) == set(hits["wafer_id"])
    assert hits["chamber"].nunique() == 3
    gt = ground_truth(sc, fab, w, 1, genealogy)
    assert "chamber" not in gt and gt["causes"][0]["chambers"] == ["ETCH-02/A", "ETCH-02/B", "ETCH-02/C"]


def test_two_causes_first_listed_wins(tmp_path, fab, genealogy, big_pool):
    a = _cause("chamber", "ETCH", "ETCH-03/B", 12, 18, "Edge-Ring", 1.0)
    b = _cause("chamber", "CMP", "CMP-01/A", 10, 22, "Center", 1.0)
    sc = _scenario(tmp_path, fab, a + b)
    w, stats = build_wafers(genealogy, sc, fab, big_pool, seed=1, return_stats=True)
    both = stats[0]["affected"] & stats[1]["affected"]
    assert both.any()
    assert (w.loc[both, "true_pattern"] == "Edge-Ring").all()  # 강도 1.0이면 먼저 원인이 모두 가져간다
    only_b = stats[1]["affected"] & ~stats[0]["affected"]
    assert (w.loc[only_b, "true_pattern"] == "Center").all()
    gt = ground_truth(sc, fab, w, 1, genealogy)
    assert [c["location"] for c in gt["causes"]] == ["ETCH-03/B", "CMP-01/A"]
    assert set(gt["affected_lots"]) == set(gt["causes"][0]["affected_lots"]) | set(gt["causes"][1]["affected_lots"])


def test_linear_intensity_rises_over_window(tmp_path, fab, genealogy, big_pool):
    sc = _scenario(tmp_path, fab, _cause("tool", "ETCH", "ETCH-02", 6, 30, "Edge-Ring", "{start: 0.0, end: 1.0}"))
    w, stats = build_wafers(genealogy, sc, fab, big_pool, seed=3, return_stats=True)
    passes = cause_passes(genealogy, sc.causes[0], fab).set_index("wafer_id")["ts"]
    hit = w.loc[stats[0]["affected"]].set_index("wafer_id")
    early = hit.loc[passes.loc[hit.index] < passes.quantile(0.33), "injected"].mean()
    late = hit.loc[passes.loc[hit.index] > passes.quantile(0.67), "injected"].mean()
    assert early < 0.4 < 0.6 < late


def test_no_cause_injects_nothing(tmp_path, fab, genealogy, pool):
    path = tmp_path / "s5.yaml"
    path.write_text("id: S5\ncauses: []", encoding="utf-8")
    sc = load_scenario(path, fab)
    w = build_wafers(genealogy, sc, fab, pool, seed=1)
    assert not w["affected"].any() and not w["injected"].any()
    gt = ground_truth(sc, fab, w, 1, genealogy)
    assert gt["causes"] == [] and gt["affected_lots"] == [] and "chamber" not in gt
