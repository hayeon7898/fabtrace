"""원인 목록 정답에 대한 일반 채점 (스펙 05 범위 B). 정답은 여기서만 쓴다.

원인이 여럿이거나, 설비 단위이거나, 없는 시나리오도 같은 기준으로 채점한다. 원인이 챔버 하나인 S1에서는
원인 특정과 격리 값이 pipeline.score_result와 같다. score_result는 바꾸지 않는다(E3, E4 결과 보존).

규칙
- 지목한 원인: 모든 패턴의 모든 Finding이 낸 (패턴, 챔버). 설비 이름을 지목하면 그 설비의 챔버 전부로 본다.
- 원인 대조: 지목한 챔버가 같은 패턴의 정답 원인 챔버에 속하면 맞다. 패턴이 다르면 맞지 않다.
- 감지: 원인 패턴의 Finding이 원인 기간과 겹치면(끝은 2일 여유, score_result와 같음) 감지로 본다.
  그런 Finding을 그 원인의 hit라고 부른다.
- 원인 특정: hit Finding이 그 원인의 챔버를 모두 지목해야 찾은 것이다. 설비 단위 원인에서 챔버 일부만
  지목하면 못 찾은 것으로 보고, 지목한 비율은 chamber_fraction으로 남긴다. 챔버 하나만 원인으로 보면
  나머지 챔버를 지난 로트를 보류하지 않기 때문이다.
- 진짜가 아닌 원인: 지목한 (패턴, 챔버) 중 같은 패턴의 어느 정답 원인에도 속하지 않는 것. E4의
  wrong_causes_all과 같은 정의다.
- 원인별 격리: hit Finding의 격리 중 그 원인 챔버의 것을 모두 합친 보류 로트로 lot recall, precision,
  wafer recall, 맞춰진 추가 보류(extra_lots_matched)를 낸다. 한 원인에 이상 구간이 둘 이상 나오면 QE는
  모두 보류하므로 합친다.
- 첫 구간 기준(*_first): score_result와 E3, E4는 첫 hit Finding만 보았다. 비교를 위해 첫 hit Finding만으로
  낸 원인 특정과 격리 값도 함께 낸다. S1에서 이 값들이 score_result와 같다.
- 전체 격리: 모든 Finding의 격리를 합친 실제 보류 전체로 lot recall과 추가 보류(extra_lots_total)를
  낸다. 진짜가 아닌 원인 때문에 생긴 보류 부담이 여기에 드러난다.
- 원인 없음(S5): 원인을 하나도 지목하지 않으면 정답이다. 원인 recall은 정의하지 않는다(None).
"""

import pandas as pd

from fabtrace.analysis.commonality import rank_of

DETECT_SLACK_DAYS = 2  # score_result와 같은 감지 여유


def _hits(result, cause: dict) -> list:
    lo, hi = cause["window"]["start_day"], cause["window"]["end_day"]
    return [f for f in result.findings
            if f.pattern == cause["pattern"] and f.window[0] <= hi + DETECT_SLACK_DAYS and f.window[1] >= lo]


def _expand(name: str, causes: list[dict]) -> set[str]:
    """설비 이름이면 그 설비의 챔버 전부, 챔버면 그대로."""
    for c in causes:
        if c["unit"] == "tool" and c["location"] == name:
            return set(c["chambers"])
    return {name}


def _affected_wafers(cause: dict, genealogy: pd.DataFrame) -> set[str]:
    lo, hi = pd.Timestamp(cause["window"]["start_ts"]), pd.Timestamp(cause["window"]["end_ts"])
    hit = genealogy["chamber"].isin(cause["chambers"]) & (genealogy["ts"] >= lo) & (genealogy["ts"] < hi)
    return set(genealogy.loc[hit, "wafer_id"])


def _pr(found: set, truth: set) -> tuple[float, float]:
    tp = len(found & truth)
    return (tp / len(found) if found else 0.0), (tp / len(truth) if truth else 0.0)


def _holds(hits: list, chambers: set, true_lots: set, true_wafers: set):
    """hit Finding들의 격리 중 원인 챔버의 것을 합친 (보류 로트, lot precision, lot recall, wafer recall)."""
    held = [k for f in hits for k in f.containments if k.chamber in chambers]
    lots = set().union(*(k.lots for k in held)) if held else set()
    wafers = set().union(*(set(k.wafers["wafer_id"]) for k in held)) if held else set()
    lot_p, lot_r = _pr(lots, true_lots)
    return lots, lot_p, lot_r, _pr(wafers, true_wafers)[1]


def score_general(result, ground_truth: dict, wafers: pd.DataFrame, genealogy: pd.DataFrame) -> dict:
    """시나리오 전체 점수와 원인별 점수. 원인별 점수는 out["causes"]에 정답 원인 순서대로 있다."""
    causes = ground_truth["causes"]
    named = {(f.pattern, ch) for f in result.findings for c in f.causes for ch in _expand(c, causes)}
    correct = {(p, ch) for p, ch in named if any(p == c["pattern"] and ch in c["chambers"] for c in causes)}
    wrong = sorted(named - correct)

    per_cause = []
    for c in causes:
        chambers = set(c["chambers"])
        hits = _hits(result, c)
        hit_named = {ch for f in hits for x in f.causes for ch in _expand(x, causes)} & chambers
        fraction = len(hit_named) / len(chambers)
        ranks = [rank_of(hits[0].ranking_b, ch) for ch in chambers] if hits else []
        true_lots, true_wafers = set(c["affected_lots"]), _affected_wafers(c, genealogy)
        lots, lot_p, lot_r, wafer_r = _holds(hits, chambers, true_lots, true_wafers)
        first = _holds(hits[:1], chambers, true_lots, true_wafers)
        first_named = {ch for x in (hits[0].causes if hits else []) for ch in _expand(x, causes)} & chambers
        per_cause.append({
            "location": c["location"], "unit": c["unit"], "pattern": c["pattern"],
            "detected": bool(hits),
            "window": list(hits[0].window) if hits else None,
            "rank_b": min(ranks) if ranks else None,
            "chamber_fraction": fraction,
            "found": fraction == 1.0,
            "held_lots": len(lots), "lot_precision": lot_p, "lot_recall": lot_r, "wafer_recall": wafer_r,
            "missed_lots": sorted(true_lots - lots),
            "extra_lots_matched": len(lots - true_lots),
            "found_first": bool(chambers) and first_named == chambers,
            "held_lots_first": len(first[0]), "lot_precision_first": first[1], "lot_recall_first": first[2],
            "wafer_recall_first": first[3],
        })

    all_held = set().union(*(k.lots for f in result.findings for k in f.containments)) \
        if any(f.containments for f in result.findings) else set()
    all_true = set(ground_truth["affected_lots"])
    n = len(causes)
    return {
        "n_causes": n,
        "causes_found": sum(c["found"] for c in per_cause),
        "cause_recall": sum(c["found"] for c in per_cause) / n if n else None,
        "named_causes": len(named),
        "cause_precision": len(correct) / len(named) if named else None,
        "wrong_causes": wrong,
        "no_cause_correct": (len(named) == 0) if n == 0 else None,
        "held_lots_total": len(all_held),
        "lot_recall_total": _pr(all_held, all_true)[1] if all_true else None,
        "extra_lots_total": len(all_held - all_true),
        "causes": per_cause,
    }


def score_row(result, ground_truth: dict, wafers: pd.DataFrame, genealogy: pd.DataFrame) -> dict:
    """csv 한 줄용으로 펼친 점수. 원인별 값은 c0_, c1_ 접두사를 붙인다."""
    s = score_general(result, ground_truth, wafers, genealogy)
    row = {k: v for k, v in s.items() if k not in ("causes", "wrong_causes")}
    row["n_wrong_causes"] = len(s["wrong_causes"])
    row["wrong_causes"] = ";".join(f"{p}:{ch}" for p, ch in s["wrong_causes"])
    for i, c in enumerate(s["causes"]):
        for k in ("location", "detected", "found", "chamber_fraction", "rank_b", "lot_recall", "lot_precision",
                  "wafer_recall", "held_lots", "extra_lots_matched", "found_first", "lot_recall_first"):
            row[f"c{i}_{k}"] = c[k]
    return row
