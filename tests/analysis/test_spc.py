import numpy as np
import pandas as pd
import pytest

from fabtrace.analysis.spc import daily_counts, detect, excursions, p_chart, plot_p_chart


def synthetic_counts(seed: int, shift: float | None = None, days=range(1, 41), n=200, base=0.054):
    """날짜별 n장, 평상시 비율 base. shift가 있으면 12~18일차 비율을 shift로 올린다."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        p = shift if shift is not None and 12 <= d <= 18 else base
        rows.append({"day": d, "n": n, "defects": int(rng.binomial(n, p))})
    out = pd.DataFrame(rows).set_index("day")
    out["p"] = out["defects"] / out["n"]
    return out


def test_limits_widen_for_small_samples():
    counts = synthetic_counts(0)
    counts.loc[5, "n"] = 20
    counts["p"] = counts["defects"] / counts["n"]
    chart = p_chart(counts)
    assert chart.loc[5, "ucl3"] > chart.loc[6, "ucl3"]


def test_center_is_not_inflated_by_excursion():
    chart = p_chart(synthetic_counts(1, shift=0.20))
    assert abs(chart["center"].iloc[0] - 0.054) < 0.01


def test_finds_shifted_window():
    windows = excursions(p_chart(synthetic_counts(2, shift=0.095)))
    assert any(s <= 18 and e >= 12 for s, e in windows)


def test_rule2_catches_more_than_rule1_on_gradual_shift():
    found_rule1 = found_any = 0
    for seed in range(20):
        chart = p_chart(synthetic_counts(seed, shift=0.095)).loc[12:18]
        found_rule1 += int(chart["rule1"].sum())
        found_any += int(chart["alarm"].sum())
    assert found_any > found_rule1


def test_few_false_alarms_without_shift():
    total = sum(len(excursions(p_chart(synthetic_counts(seed)))) for seed in range(20))
    assert total <= 6  # 20개 데이터, 각 40일 기준 오탐 구간 합계


def test_excursions_merge_one_day_gap():
    chart = pd.DataFrame({"alarm": [False, True, False, True, False, False, True]}, index=range(1, 8))
    assert excursions(chart) == [(2, 4), (7, 7)]


def test_requires_predictions_not_truth():
    truth = pd.DataFrame({"wafer_id": ["a"], "true_pattern": ["Edge-Ring"]})
    with pytest.raises(ValueError):
        daily_counts(truth, pd.Series({"a": 1}, name="day"), "Edge-Ring")


def test_detect_end_to_end(genealogy, fab, s1, tmp_path):
    """가상 팹 이력 위에서 주입 기간 웨이퍼만 예측 Edge-Ring 비율을 올리면 그 근처를 찾는다."""
    rng = np.random.default_rng(0)
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    lo, hi = s1.window(fab.start_date)
    etch = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    hit = wafers["wafer_id"].isin(etch["wafer_id"])
    p = np.where(hit, 0.45, 0.054)
    wafers["pred_pattern"] = np.where(rng.random(len(wafers)) < p, "Edge-Ring", "none")

    chart, windows = detect(wafers, genealogy, "Edge-Ring", fab.start_date)
    assert any(s <= 19 and e >= 12 for s, e in windows)
    plot_p_chart(chart, windows, "Edge-Ring", tmp_path / "p.png")
    assert (tmp_path / "p.png").exists()