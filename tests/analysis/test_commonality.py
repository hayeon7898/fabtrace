import numpy as np
import pandas as pd

from fabtrace.analysis.commonality import analyze, chamber_table, method_a, method_b, rank_of


def synthetic(seed: int, cause: str | None = "ET-02/B", n: int = 3000, base: float = 0.05, hit: float = 0.30):
    """공정 3단계. 마지막 단계는 모든 웨이퍼가 거치는 병목. cause를 거친 웨이퍼는 불량률이 hit."""
    rng = np.random.default_rng(seed)
    steps = {
        "DEP": ["DP-01/A", "DP-01/B", "DP-02/A", "DP-02/B"],
        "ETCH": ["ET-01/A", "ET-01/B", "ET-01/C", "ET-02/A", "ET-02/B", "ET-02/C"],
        "CLEAN": ["CL-01/A"],
    }
    rows, defects = [], {}
    for i in range(n):
        wid = f"W{i:05d}"
        passed = [rng.choice(chs) for chs in steps.values()]
        for (step, _), ch in zip(steps.items(), passed):
            rows.append({"wafer_id": wid, "step": step, "tool": ch.split("/")[0], "chamber": ch})
        p = hit if cause in passed else base
        defects[wid] = rng.random() < p
    gen = pd.DataFrame(rows)
    pop = pd.DataFrame({"wafer_id": list(defects), "day": 1, "is_defect": list(defects.values())})
    return gen, pop


def test_counts_are_consistent():
    gen, pop = synthetic(0)
    t = chamber_table(pop, gen)
    assert (t["passed"] + t["not_passed"] == len(pop)).all()
    assert (t["defect_passed"] + t["defect_not_passed"] == pop["is_defect"].sum()).all()


def test_method_a_falls_for_bottleneck_and_b_finds_cause():
    gen, pop = synthetic(1)
    t = chamber_table(pop, gen)
    assert method_a(t).iloc[0]["chamber"] == "CL-01/A"
    b = method_b(t)
    assert b.iloc[0]["chamber"] == "ET-02/B"
    assert b.iloc[0]["significant"]


def test_bottleneck_is_never_significant():
    gen, pop = synthetic(2)
    b = method_b(chamber_table(pop, gen))
    row = b[b["chamber"] == "CL-01/A"].iloc[0]
    assert row["p"] == 1.0 and not row["significant"]


def test_no_cause_few_significant():
    hits = 0
    for seed in range(20):
        gen, pop = synthetic(seed, cause=None)
        hits += int(method_b(chamber_table(pop, gen))["significant"].sum())
    assert hits <= 2  # 20개 데이터, 각 11개 챔버 검정 기준


def test_analyze_on_virtual_fab(genealogy, fab, s1):
    """가상 팹 이력에서 원인 챔버를 지난 웨이퍼의 예측 불량률만 올리면 방법 B가 찾는다."""
    rng = np.random.default_rng(0)
    wafers = genealogy[["wafer_id"]].drop_duplicates().reset_index(drop=True)
    lo, hi = s1.window(fab.start_date)
    etch = genealogy[(genealogy["chamber"] == s1.chamber) & genealogy["ts"].between(lo, hi, inclusive="left")]
    p = np.where(wafers["wafer_id"].isin(etch["wafer_id"]), 0.45, 0.054)
    wafers["pred_pattern"] = np.where(rng.random(len(wafers)) < p, "Edge-Ring", "none")

    res = analyze(wafers, genealogy, (13, 18), fab.start_date, "Edge-Ring")
    assert rank_of(res["B"], s1.chamber) == 1
    assert res["A"].iloc[0]["chamber"] == "CLN-01/A"