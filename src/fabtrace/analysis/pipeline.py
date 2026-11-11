"""원인 분석 파이프라인: 관리도 감시 -> 공통성 분석 -> 격리 범위. 그리고 정답지로 채점.

- run_analysis는 분류기 예측(pred_pattern)과 공정 이력만 받는다. 정답은 score_result만 읽는다.
- 불량 패턴 8개를 모두 관리도로 감시하고, 이상 구간이 나온 패턴만 공통성 분석을 한다.
- 방법 B에서 유의한 챔버를 모두 원인 후보로 보고 각각 격리 범위를 정한다. 없으면 원인 미확정.
- simulate_and_predict는 시드로 가상 팹을 만들고 분류기로 예측한다. E3가 시드만 바꿔 재사용한다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fabtrace.analysis.commonality import analyze, rank_of
from fabtrace.analysis.containment import DEFAULT_BUFFER_HOURS, Containment, contain, score
from fabtrace.analysis.spc import PRED_COL, detect
from fabtrace.data.wm811k import CLASSES

DEFECT_PATTERNS = [c for c in CLASSES if c != "none"]


@dataclass
class Finding:
    pattern: str
    window: tuple[int, int]
    population: int
    defects: int
    ranking_a: pd.DataFrame
    ranking_b: pd.DataFrame
    causes: list[str]
    containments: list[Containment]


@dataclass
class RCAResult:
    start_date: str
    buffer_hours: float
    charts: dict[str, pd.DataFrame] = field(default_factory=dict)
    windows: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def run_analysis(predictions: pd.DataFrame, genealogy: pd.DataFrame, start_date: str,
                 patterns: list[str] = DEFECT_PATTERNS,
                 buffer_hours: float = DEFAULT_BUFFER_HOURS) -> RCAResult:
    preds = predictions[["wafer_id", PRED_COL]]  # 예측 외의 컬럼은 들여오지 않는다
    result = RCAResult(start_date=start_date, buffer_hours=buffer_hours)
    for pattern in patterns:
        chart, windows = detect(preds, genealogy, pattern, start_date)
        result.charts[pattern], result.windows[pattern] = chart, windows
        for window in windows:
            res = analyze(preds, genealogy, window, start_date, pattern)
            b = res["B"]
            causes = b.loc[b["significant"], "chamber"].tolist()
            result.findings.append(Finding(
                pattern=pattern, window=window,
                population=len(res["population"]), defects=int(res["population"]["is_defect"].sum()),
                ranking_a=res["A"], ranking_b=b, causes=causes,
                containments=[contain(res["population"], genealogy, ch, buffer_hours) for ch in causes],
            ))
    return result


def score_result(result: RCAResult, ground_truth: dict, wafers: pd.DataFrame) -> dict:
    """정답지와 비교한 채점. 정답은 여기서만 쓴다."""
    pattern, chamber = ground_truth["pattern"], ground_truth["chamber"]
    lo, hi = ground_truth["window"]["start_day"], ground_truth["window"]["end_day"]
    hits = [f for f in result.findings if f.pattern == pattern and f.window[0] <= hi + 2 and f.window[1] >= lo]
    other_alarms = sum(len(w) for p, w in result.windows.items() if p != pattern)
    out = {"detected": bool(hits), "false_alarm_windows_other_patterns": other_alarms}
    if not hits:
        return out | {"top1_a": False, "top1_b": False, "top3_b": False, "cause_found": False,
                      "lot_precision": 0.0, "lot_recall": 0.0}
    f = hits[0]
    rank_a, rank_b = rank_of(f.ranking_a, chamber), rank_of(f.ranking_b, chamber)
    out |= {
        "window": list(f.window),
        "rank_a": rank_a, "rank_b": rank_b,
        "top1_a": rank_a == 1, "top1_b": rank_b == 1, "top3_b": rank_b <= 3,
        "bottleneck_top1_a": bool(f.ranking_a.iloc[0]["passed"] == f.population),
        "causes": f.causes,
        "cause_found": chamber in f.causes,
        "false_causes": [c for c in f.causes if c != chamber],
    }
    match = next((c for c in f.containments if c.chamber == chamber), None)
    if match is None:
        return out | {"lot_precision": 0.0, "lot_recall": 0.0}
    s = score(match, ground_truth["affected_lots"], wafers.loc[wafers["affected"], "wafer_id"])
    return out | {k: s[k] for k in ("held_lots", "lot_precision", "lot_recall", "missed_lots",
                                     "extra_lots", "wafer_precision", "wafer_recall")}


def simulate_and_predict(scenario_path: Path, fab_path: Path, seed: int, data, model_path: Path):
    """시드로 가상 팹을 만들고 분류기로 예측한다. (genealogy, wafers, ground_truth, predictions)"""
    from fabtrace.models.evaluate import predict_wafers
    from fabtrace.sim.config import load_fab_config
    from fabtrace.sim.genealogy import generate_genealogy
    from fabtrace.sim.inject import build_wafers, ground_truth
    from fabtrace.sim.scenario import load_scenario

    fab = load_fab_config(fab_path)
    sc = load_scenario(scenario_path, fab)
    fab_idx = data.indices("fab")
    pool = pd.DataFrame({"label": [CLASSES[i] for i in data.labels[fab_idx]]}, index=fab_idx)

    genealogy = generate_genealogy(fab, seed)
    wafers = build_wafers(genealogy, sc, fab, pool, seed)
    gt = ground_truth(sc, fab, wafers, seed)
    predictions = predict_wafers(model_path, data, wafers)
    return fab, genealogy, wafers, gt, predictions