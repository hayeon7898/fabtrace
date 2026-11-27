"""configs/scenarios의 S1~S5가 의도대로 주입되는지 확인한다 (#47)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fabtrace.sim.inject import build_wafers, cause_passes
from fabtrace.sim.scenario import INTENSITY_SWEEP, load_scenario, with_intensity

SCENARIOS = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
# 실제 WM-811K 라벨 비율. 평상시 패턴 분포로 쓴다.
WM811K = {"none": 147431, "Edge-Ring": 9680, "Edge-Loc": 5189, "Center": 4294, "Loc": 3593,
          "Scratch": 1193, "Random": 866, "Donut": 555, "Near-full": 149}


@pytest.fixture(scope="module")
def pool():
    labels = [c for c, n in WM811K.items() for _ in range(n)]
    return pd.DataFrame({"label": labels}, index=range(len(labels)))


def _load(fab, name):
    return load_scenario(SCENARIOS / f"{name}.yaml", fab)


def _rate(w, mask):
    return w.loc[mask, "injected"].mean(), int(mask.sum())


@pytest.mark.parametrize("name,n_causes", [("s1", 1), ("s2", 1), ("s3", 2), ("s4", 1), ("s5", 0)])
def test_scenario_files_load(fab, name, n_causes):
    assert len(_load(fab, name).causes) == n_causes


def test_s2_tool_cause_injects_at_its_intensity(fab, genealogy, pool):
    sc = _load(fab, "s2")
    w, stats = build_wafers(genealogy, sc, fab, pool, seed=1, return_stats=True)
    rate, n = _rate(w, stats[0]["affected"])
    assert n > 300 and abs(rate - 0.2) < 0.06
    chambers = cause_passes(genealogy, sc.causes[0], fab)["chamber"]
    assert set(chambers) == {"ETCH-02/A", "ETCH-02/B", "ETCH-02/C"}


def test_s3_second_cause_keeps_its_intensity_outside_overlap(fab, genealogy, pool):
    w, stats = build_wafers(genealogy, _load(fab, "s3"), fab, pool, seed=1, return_stats=True)
    a, b = stats[0]["affected"], stats[1]["affected"]
    assert (a & b).any()  # 두 원인에 함께 걸린 웨이퍼가 있다
    rate_a, _ = _rate(w, a)
    rate_b, n_b = _rate(w, b & ~a)
    assert abs(rate_a - 0.4) < 0.1 and n_b > 50 and abs(rate_b - 0.4) < 0.12
    assert (w.loc[w["injected"], "true_pattern"] == "Edge-Ring").all()


def test_s4_linear_intensity_averages_half(fab, genealogy, pool):
    sc = _load(fab, "s4")
    w, stats = build_wafers(genealogy, sc, fab, pool, seed=1, return_stats=True)
    hit = stats[0]["affected"]
    rate, n = _rate(w, hit)
    assert n > 100 and abs(rate - 0.2) < 0.07  # 0에서 0.4까지 선형이면 평균 0.2
    passes = cause_passes(genealogy, sc.causes[0], fab).set_index("wafer_id")["ts"]
    ts = passes.loc[w.loc[hit, "wafer_id"]].astype("int64").to_numpy()
    inj = w.loc[hit, "injected"].to_numpy()
    mid = np.median(ts)
    assert inj[ts < mid].mean() < inj[ts >= mid].mean()


def test_s5_follows_natural_mix(fab, genealogy, pool):
    w = build_wafers(genealogy, _load(fab, "s5"), fab, pool, seed=1)
    assert not w["affected"].any() and not w["injected"].any()
    observed = w["true_pattern"].value_counts(normalize=True)
    expected = pool["label"].value_counts(normalize=True)
    assert (observed.reindex(expected.index, fill_value=0) - expected).abs().max() < 0.01


@pytest.mark.parametrize("intensity", INTENSITY_SWEEP)
def test_intensity_sweep_on_s2(fab, genealogy, pool, intensity):
    sc = with_intensity(_load(fab, "s2"), intensity)
    assert sc.id == f"S2@{intensity:g}"
    w, stats = build_wafers(genealogy, sc, fab, pool, seed=2, return_stats=True)
    rate, _ = _rate(w, stats[0]["affected"])
    assert abs(rate - intensity) < 0.07


def test_intensity_sweep_rejects_linear(fab):
    with pytest.raises(ValueError):
        with_intensity(_load(fab, "s4"), 0.2)
