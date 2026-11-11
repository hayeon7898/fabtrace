"""E3: 여러 시드에서 원인 분석 성능을 잰다.

시드마다 가상 팹(이력, 맵 배정)을 새로 만들고, 분류기는 cnn_v1로 고정한다.
seed 42는 규칙과 파라미터를 다듬는 데 쓴 개발용 시드라 평가에서 제외하고 참고로만 보인다.

시드 하나에서 재는 것
- 감지: Edge-Ring 이상 구간을 찾았는가, rule1만으로도 찾았는가, rule2가 더 잡은 날
- 원인: 방법 A와 B에서 원인 챔버 순위, 병목이 방법 A 1위인가, 잘못 지목한 챔버 수
- 격리: 여유 0, 12, 24시간별 로트 precision, recall, 웨이퍼 recall
- 오탐: 다른 패턴의 이상 구간 수
"""

import numpy as np
import pandas as pd

from fabtrace.analysis.commonality import window_population
from fabtrace.analysis.containment import contain, score
from fabtrace.analysis.pipeline import run_analysis, score_result
from fabtrace.analysis.spc import excursions

BUFFERS = (0, 12, 24)


def evaluate_seed(seed: int, genealogy: pd.DataFrame, wafers: pd.DataFrame, gt: dict,
                  predictions: pd.DataFrame, start_date: str) -> dict:
    result = run_analysis(predictions, genealogy, start_date, buffer_hours=24)
    ev = score_result(result, gt, wafers)
    pattern, chamber = gt["pattern"], gt["chamber"]
    lo, hi = gt["window"]["start_day"], gt["window"]["end_day"]

    chart = result.charts[pattern]
    near = chart.loc[(chart.index >= lo) & (chart.index <= hi + 2)]
    rule1_windows = excursions(chart.assign(alarm=chart["rule1"]))
    row = {
        "seed": seed,
        "detected": ev["detected"],
        "detected_rule1_only": any(s <= hi + 2 and e >= lo for s, e in rule1_windows),
        "rule1_days": int(near["rule1"].sum()),
        "rule2_extra_days": int((near["rule2"] & ~near["rule1"]).sum()),
        "window": "-".join(map(str, ev.get("window", []))),
        "rank_a": ev.get("rank_a"),
        "rank_b": ev.get("rank_b"),
        "bottleneck_top1_a": ev.get("bottleneck_top1_a", False),
        "top1_a": ev["top1_a"],
        "top1_b": ev["top1_b"],
        "top3_b": ev["top3_b"],
        "cause_found": ev["cause_found"],
        "false_causes": len(ev.get("false_causes", [])),
        "other_pattern_alarms": ev["false_alarm_windows_other_patterns"],
        "affected_lots": len(gt["affected_lots"]),
    }

    truth_wafers = wafers.loc[wafers["affected"], "wafer_id"]
    finding = next((f for f in result.findings
                    if f.pattern == pattern and f.window[0] <= hi + 2 and f.window[1] >= lo), None)
    for b in BUFFERS:
        if finding is None or chamber not in finding.causes:
            row |= {f"lot_precision_{b}h": 0.0, f"lot_recall_{b}h": 0.0, f"wafer_recall_{b}h": 0.0, f"held_lots_{b}h": 0}
            continue
        pop = window_population(predictions, genealogy, finding.window, start_date, pattern)
        s = score(contain(pop, genealogy, chamber, b), gt["affected_lots"], truth_wafers)
        row |= {f"lot_precision_{b}h": s["lot_precision"], f"lot_recall_{b}h": s["lot_recall"],
                f"wafer_recall_{b}h": s["wafer_recall"], f"held_lots_{b}h": s["held_lots"]}
    return row


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    out = {"seeds": n}
    for k in ("detected", "detected_rule1_only", "bottleneck_top1_a", "top1_a", "top1_b", "top3_b", "cause_found"):
        out[k] = f"{int(df[k].sum())}/{n}"
    out["seeds_with_rule2_extra_days"] = f"{int((df['rule2_extra_days'] > 0).sum())}/{n}"
    out["false_causes_total"] = int(df["false_causes"].sum())
    out["other_pattern_alarms_total"] = int(df["other_pattern_alarms"].sum())
    out["rank_a_median"] = float(np.nanmedian(df["rank_a"].astype(float)))
    for b in BUFFERS:
        for m in ("lot_precision", "lot_recall", "wafer_recall"):
            col = df[f"{m}_{b}h"]
            out[f"{m}_{b}h"] = f"{col.mean():.3f} (최소 {col.min():.3f})"
        out[f"seeds_missing_lots_{b}h"] = f"{int((df[f'lot_recall_{b}h'] < 1).sum())}/{n}"
    return out