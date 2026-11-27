"""원인 분석 파이프라인: 관리도 감시 -> 공통성 분석 -> 격리 범위. 그리고 정답지로 채점.

- run_analysis는 분류기 예측(pred_pattern)과 공정 이력만 받는다. 정답은 score_result만 읽는다.
- 불량 패턴 8개를 모두 관리도로 감시하고, 이상 구간이 나온 패턴만 공통성 분석을 한다.
- 방법 B에서 유의한 챔버를 모두 원인 후보로 보고 각각 격리 범위를 정한다. 없으면 원인 미확정.
- simulate_and_predict는 시드로 가상 팹을 만들고 분류기로 예측한다. E3가 시드만 바꿔 재사용한다.
- method로 격리 범위를 정하는 방법을 고른다. E4가 같은 입력에 방법만 바꿔 비교한다.
  - baseline: 이상 구간 웨이퍼가 원인 챔버를 지난 시각으로 노출 기간을 정한다 (E3와 같음).
  - exposure: 원인 챔버의 통과 날짜별 관리도로 노출 기간을 다시 잡는다 (스펙 04 범위 A).
  - exposure+chamber: exposure에 챔버 단위 감시를 더한다 (스펙 04 범위 B).
  감시와 공통성 분석은 baseline과 exposure가 같다. exposure는 격리 범위만 바꾼다.
- exposure+chamber는 공장 전체 감시가 끝난 뒤 챔버 감시를 돌린다. 알림을 팹 아웃 구간으로 바꿔 공통성
  분석으로 확인하고(chamber_monitor.confirm_alarm), 알린 날이 2일 이상이고 확인된 알림만 source="chamber"인 Finding으로 더한다.
  공장 전체 Finding이 같은 챔버를 원인으로 이미 냈고 구간이 겹치면 건너뛴다. 챔버 Finding은 항상
  공장 전체 Finding 뒤에 붙으므로, 공장 전체가 찾은 결과와 채점은 exposure와 같다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fabtrace.analysis.chamber_monitor import (MIN_ALARM_DAYS, ChamberMonitor, alarm_windows, confirm_alarm,
                                              monitor)
from fabtrace.analysis.commonality import analyze, rank_of
from fabtrace.analysis.containment import DEFAULT_BUFFER_HOURS, Containment, contain, score
from fabtrace.analysis.exposure import ChamberExposure, contain_by_chamber
from fabtrace.analysis.spc import PRED_COL, detect
from fabtrace.data.wm811k import CLASSES

DEFECT_PATTERNS = [c for c in CLASSES if c != "none"]
METHODS = ("baseline", "exposure", "exposure+chamber")
DEFAULT_METHOD = "baseline"


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
    exposures: list[ChamberExposure] = field(default_factory=list)  # exposure 모드에서만, containments와 같은 순서
    source: str = "fab"          # fab: 공장 전체 관리도, chamber: 챔버 감시 알림
    alarm: dict | None = None    # source가 chamber일 때 확인한 알림 (chamber_monitor.WINDOW_COLUMNS)


@dataclass
class RCAResult:
    start_date: str
    buffer_hours: float
    method: str = DEFAULT_METHOD
    charts: dict[str, pd.DataFrame] = field(default_factory=dict)
    windows: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    chamber_monitor: ChamberMonitor | None = None  # exposure+chamber에서만
    chamber_alarms: pd.DataFrame | None = None     # 알림별 팹 아웃 구간과 처리 결과(status)


def check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"알 수 없는 방법 {method!r}. 가능한 값: {', '.join(METHODS)}")


def contain_finding(population: pd.DataFrame, predictions: pd.DataFrame, genealogy: pd.DataFrame,
                    chamber: str, pattern: str, start_date: str, buffer_hours: float,
                    method: str = DEFAULT_METHOD) -> tuple[Containment, ChamberExposure | None]:
    """원인 후보 하나의 격리 범위. 방법 선택은 여기 한 곳에서만 한다."""
    check_method(method)
    if method == "baseline":
        return contain(population, genealogy, chamber, buffer_hours), None
    return contain_by_chamber(population, predictions, genealogy, chamber, pattern, start_date, buffer_hours)


def run_analysis(predictions: pd.DataFrame, genealogy: pd.DataFrame, start_date: str,
                 patterns: list[str] = DEFECT_PATTERNS,
                 buffer_hours: float = DEFAULT_BUFFER_HOURS,
                 method: str = DEFAULT_METHOD) -> RCAResult:
    check_method(method)
    preds = predictions[["wafer_id", PRED_COL]]  # 예측 외의 컬럼은 들여오지 않는다
    result = RCAResult(start_date=start_date, buffer_hours=buffer_hours, method=method)
    for pattern in patterns:
        chart, windows = detect(preds, genealogy, pattern, start_date)
        result.charts[pattern], result.windows[pattern] = chart, windows
        for window in windows:
            res = analyze(preds, genealogy, window, start_date, pattern)
            b = res["B"]
            causes = b.loc[b["significant"], "chamber"].tolist()
            contained = [contain_finding(res["population"], preds, genealogy, ch, pattern, start_date,
                                         buffer_hours, method) for ch in causes]
            result.findings.append(_finding(pattern, window, res, causes, contained))
    if method == "exposure+chamber":
        _add_chamber_findings(result, preds, genealogy, patterns)
    return result


def _finding(pattern, window, res, causes, contained, **extra) -> Finding:
    return Finding(
        pattern=pattern, window=window,
        population=len(res["population"]), defects=int(res["population"]["is_defect"].sum()),
        ranking_a=res["A"], ranking_b=res["B"], causes=causes,
        containments=[c for c, _ in contained],
        exposures=[e for _, e in contained if e is not None], **extra,
    )


def _covered(findings: list[Finding], pattern: str, chamber: str, window: tuple[int, int]) -> bool:
    """같은 패턴에서 이 챔버를 원인으로 이미 낸 Finding이 있고 구간이 겹치는가."""
    return any(f.pattern == pattern and chamber in f.causes and f.window[0] <= window[1] and f.window[1] >= window[0]
               for f in findings)


def _add_chamber_findings(result: RCAResult, preds: pd.DataFrame, genealogy: pd.DataFrame,
                          patterns: list[str]) -> None:
    centers = {p: float(result.charts[p]["center"].iloc[0]) for p in patterns}
    result.chamber_monitor = monitor(preds, genealogy, result.start_date, patterns, centers=centers)
    alarms = alarm_windows(result.chamber_monitor.alarms, genealogy, result.start_date)
    status = []
    for a in alarms.to_dict("records"):
        pattern, chamber, window = a["pattern"], a["chamber"], (a["fab_out_start"], a["fab_out_end"])
        if _covered(result.findings, pattern, chamber, window):
            status.append("covered")
            continue
        if a["alarm_days"] < MIN_ALARM_DAYS:
            status.append("short")
            continue
        ok, res = confirm_alarm(preds, genealogy, window, result.start_date, pattern, chamber)
        status.append("confirmed" if ok else "rejected")
        if ok:
            contained = [contain_finding(res["population"], preds, genealogy, chamber, pattern, result.start_date,
                                         result.buffer_hours, result.method)]
            result.findings.append(_finding(pattern, window, res, [chamber], contained, source="chamber", alarm=a))
    result.chamber_alarms = alarms.assign(status=status)


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
    gt = ground_truth(sc, fab, wafers, seed, genealogy)
    predictions = predict_wafers(model_path, data, wafers)
    return fab, genealogy, wafers, gt, predictions