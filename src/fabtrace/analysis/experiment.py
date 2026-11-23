"""E3: 여러 시드에서 원인 분석 성능을 잰다.

시드마다 가상 팹(이력, 맵 배정)을 새로 만들고, 분류기는 cnn_v1로 고정한다.
seed 42는 규칙과 파라미터를 다듬는 데 쓴 개발용 시드라 평가에서 제외하고 참고로만 보인다.

시드 하나에서 재는 것
- 감지: Edge-Ring 이상 구간을 찾았는가, rule1만으로도 찾았는가, rule2가 더 잡은 날
- 원인: 방법 A와 B에서 원인 챔버 순위, 병목이 방법 A 1위인가, 잘못 지목한 챔버 수
- 격리: 여유 0, 12, 24시간별 로트 precision, recall, 웨이퍼 recall
- 오탐: 다른 패턴의 이상 구간 수

E4(#37)는 평가 시드 100~109에서 같은 팹과 예측에 세 방법을 적용해 비교한다. evaluate_methods가 시드 하나,
judge_e4가 가설 판정이다. 판정 기준은 실행 전에 정했다.

E5(#41)는 E4 전에 개선 방법의 기본 여유 시간을 seed 42만으로 다시 정한다. choose_buffer가 판단 기준이다.
"""

import numpy as np
import pandas as pd

from fabtrace.analysis.commonality import window_population
from fabtrace.analysis.containment import score
from fabtrace.analysis.pipeline import DEFAULT_METHOD, contain_finding, run_analysis, score_result
from fabtrace.analysis.spc import excursions

BUFFERS = (0, 12, 24)
REVIEW_BUFFERS = (0, 6, 12, 18, 24)
CURRENT_BUFFER = 24
# E5 판단 기준. 결과를 보기 전에 #41 이슈에 적은 값이다.
MIN_SYNTH_RECALL_MEAN = 0.98
MIN_SYNTH_RECALL_MIN = 0.9
# WM-811K 라벨 있는 맵의 클래스별 개수 (README). 합성 예측의 평상시 패턴 비율로 쓴다.
WM811K_CLASS_COUNTS = {
    "none": 147431, "Edge-Ring": 9680, "Edge-Loc": 5189, "Center": 4294, "Loc": 3593,
    "Scratch": 1193, "Random": 866, "Donut": 555, "Near-full": 149,
}


def evaluate_seed(seed: int, genealogy: pd.DataFrame, wafers: pd.DataFrame, gt: dict,
                  predictions: pd.DataFrame, start_date: str, method: str = DEFAULT_METHOD,
                  buffers: tuple[int, ...] = BUFFERS, return_result: bool = False):
    result = run_analysis(predictions, genealogy, start_date, buffer_hours=24, method=method)
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
    for b in buffers:
        if finding is None or chamber not in finding.causes:
            row |= {f"lot_precision_{b}h": 0.0, f"lot_recall_{b}h": 0.0, f"wafer_recall_{b}h": 0.0, f"held_lots_{b}h": 0}
            continue
        pop = window_population(predictions, genealogy, finding.window, start_date, pattern)
        c, _ = contain_finding(pop, predictions, genealogy, chamber, pattern, start_date, b, method)
        s = score(c, gt["affected_lots"], truth_wafers)
        row |= {f"lot_precision_{b}h": s["lot_precision"], f"lot_recall_{b}h": s["lot_recall"],
                f"wafer_recall_{b}h": s["wafer_recall"], f"held_lots_{b}h": s["held_lots"]}
    return (row, result) if return_result else row


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

def choose_buffer(real: dict, synthetic: pd.DataFrame, buffers: tuple[int, ...] = REVIEW_BUFFERS) -> tuple[int, pd.DataFrame]:
    """E5 판단 기준으로 기본 여유 시간을 고른다. (선택한 값, 여유 시간별 판정 표)

    실제 예측(seed 42)에서 lot recall 1.0이고, 합성 예측에서 lot recall 평균이 MIN_SYNTH_RECALL_MEAN 이상,
    최솟값이 MIN_SYNTH_RECALL_MIN 이상인 값 중 가장 짧은 값. 그 값이 CURRENT_BUFFER보다 짧지 않으면
    CURRENT_BUFFER를 유지한다.
    """
    rows = []
    for b in sorted(buffers):
        col = synthetic[f"lot_recall_{b}h"]
        rows.append({"buffer_hours": b, "real_lot_recall": real[f"lot_recall_{b}h"],
                     "real_lot_precision": real[f"lot_precision_{b}h"],
                     "synth_recall_mean": col.mean(), "synth_recall_min": col.min(),
                     "synth_precision_mean": synthetic[f"lot_precision_{b}h"].mean()})
    table = pd.DataFrame(rows)
    table["passes"] = ((table["real_lot_recall"] == 1.0) & (table["synth_recall_mean"] >= MIN_SYNTH_RECALL_MEAN)
                       & (table["synth_recall_min"] >= MIN_SYNTH_RECALL_MIN))
    ok = table.loc[table["passes"], "buffer_hours"]
    chosen = int(ok.min()) if len(ok) and ok.min() < CURRENT_BUFFER else CURRENT_BUFFER
    return chosen, table


E4_METHODS = ("baseline", "exposure", "exposure+chamber")
E4_BASE, E4_NEW = "baseline", "exposure+chamber"
E4_BUFFER = 24
# E4 판정 기준. 결과를 보기 전에 정했다. 근거는 스펙 04 E4와 실험 기록 E4의 방법 절에 있다.
E4_MAX_MISSING_SEEDS = 1   # 가설 2: 여유 24시간에서 로트를 놓친 시드 수 (10개 중)
E4_MAX_EXTRA_LOTS = 3      # 가설 3: baseline보다 추가로 보류하는 정상 로트, 사건당 평균
ALARM_STATUSES = ("confirmed", "rejected", "covered", "short")


def evaluate_methods(seed: int, genealogy: pd.DataFrame, wafers: pd.DataFrame, gt: dict,
                     predictions: pd.DataFrame, start_date: str,
                     methods: tuple[str, ...] = E4_METHODS) -> list[dict]:
    """같은 팹과 예측에 방법마다 evaluate_seed를 적용하고 E4 지표를 더한다."""
    rows = []
    for method in methods:
        row, result = evaluate_seed(seed, genealogy, wafers, gt, predictions, start_date, method, return_result=True)
        row = {"method": method} | row
        # 모든 Finding(모든 패턴, 챔버 감시 포함)에서 정답이 아닌 원인. E3의 false_causes는 첫 Finding만 센다.
        row["wrong_causes_all"] = sum(c != gt["chamber"] for f in result.findings for c in f.causes)
        for b in BUFFERS:  # 보류했지만 영향이 없는 로트 = 보류 로트 - 찾은 영향 로트
            row[f"extra_lots_{b}h"] = int(round(row[f"held_lots_{b}h"] - row[f"lot_recall_{b}h"] * row["affected_lots"]))
        alarms = result.chamber_alarms
        row["chamber_alarms"] = 0 if alarms is None else len(alarms)
        for st in ALARM_STATUSES:
            row[f"alarms_{st}"] = 0 if alarms is None else int((alarms["status"] == st).sum())
        rows.append(row)
    return rows


def summarize_e4(df: pd.DataFrame) -> pd.DataFrame:
    """방법별 요약. 한 행이 한 방법이다."""
    b = E4_BUFFER
    out = []
    for method, g in df.groupby("method", sort=False):
        det = g[g["detected"]]
        out.append({
            "method": method, "seeds": len(g),
            "detected": int(g["detected"].sum()), "cause_found": int(g["cause_found"].sum()),
            "top1_b": int(g["top1_b"].sum()),
            f"lot_recall_{b}h_detected": det[f"lot_recall_{b}h"].mean() if len(det) else float("nan"),
            f"lot_recall_{b}h_all": g[f"lot_recall_{b}h"].mean(),
            f"lot_precision_{b}h_detected": det[f"lot_precision_{b}h"].mean() if len(det) else float("nan"),
            f"seeds_missing_lots_{b}h": int((g[f"lot_recall_{b}h"] < 1).sum()),
            f"extra_lots_{b}h_detected": det[f"extra_lots_{b}h"].mean() if len(det) else float("nan"),
            "wrong_causes_all": int(g["wrong_causes_all"].sum()),
            "false_causes_first": int(g["false_causes"].sum()),
            "other_pattern_alarms": int(g["other_pattern_alarms"].sum()),
            "chamber_alarms": int(g["chamber_alarms"].sum()),
            "alarms_confirmed": int(g["alarms_confirmed"].sum()),
        })
    return pd.DataFrame(out)


def judge_e4(df: pd.DataFrame) -> list[dict]:
    """E4 가설 1~4를 실행 전에 정한 기준으로 판정한다. baseline 대 exposure+chamber, 여유 24시간."""
    b = E4_BUFFER
    base, new = (df[df["method"] == m].set_index("seed") for m in (E4_BASE, E4_NEW))
    n = len(base)
    out = []

    d_base, d_new = int(base["detected"].sum()), int(new["detected"].sum())
    if d_base == d_new == n:
        v1 = "판정 불가"
    else:
        v1 = "지지" if d_new > d_base else "기각"
    out.append({"hypothesis": 1, "verdict": v1, "evidence": f"감지 {d_base}/{n} -> {d_new}/{n}"})

    r_base = base.loc[base["detected"], f"lot_recall_{b}h"].mean()
    r_new = new.loc[new["detected"], f"lot_recall_{b}h"].mean()
    missing = int((new[f"lot_recall_{b}h"] < 1).sum())
    if pd.isna(r_base) or pd.isna(r_new):  # 한쪽이 감지한 시드가 없으면 평균을 비교할 수 없다
        v2 = "판정 불가"
    else:
        v2 = "지지" if r_new > r_base and missing <= E4_MAX_MISSING_SEEDS else "기각"
    out.append({"hypothesis": 2, "verdict": v2,
                "evidence": f"감지된 시드 lot recall {r_base:.3f} -> {r_new:.3f}, "
                            f"로트를 놓친 시드 {missing}/{n} (기준 {E4_MAX_MISSING_SEEDS} 이하)"})

    both = base.index[base["detected"] & new["detected"]]
    if len(both) == 0:
        out.append({"hypothesis": 3, "verdict": "판정 불가", "evidence": "두 방법 모두 감지한 시드 없음"})
    else:
        extra = (new.loc[both, f"extra_lots_{b}h"] - base.loc[both, f"extra_lots_{b}h"]).mean()
        out.append({"hypothesis": 3, "verdict": "지지" if extra <= E4_MAX_EXTRA_LOTS else "기각",
                    "evidence": f"두 방법 모두 감지한 시드 {len(both)}개에서 추가 보류 정상 로트 평균 {extra:+.1f}개 "
                                f"(기준 {E4_MAX_EXTRA_LOTS} 이하)"})

    w_base, w_new = int(base["wrong_causes_all"].sum()), int(new["wrong_causes_all"].sum())
    out.append({"hypothesis": 4, "verdict": "지지" if w_new <= w_base else "기각",
                "evidence": f"잘못 지목한 원인(모든 Finding) {w_base} -> {w_new}"})
    return out
