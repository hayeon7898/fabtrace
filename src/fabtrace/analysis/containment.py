"""격리 범위 산정: 원인 후보 챔버에 노출된 로트를 찾아 출하 보류 범위를 정한다.

- 노출 기간: 이상 구간(팹 아웃 기준)에 팹 아웃된 웨이퍼들이 원인 챔버를 지난 시각의 처음과 끝.
  관리도는 팹 아웃 시점에 불량을 보지만, 격리는 원인 챔버를 지난 시각 기준이어야 한다.
- 여유 시간: 노출 기간 앞뒤로 buffer_hours를 더한다. 불량 로트를 놓치는 비용이 정상 로트를
  보류하는 비용보다 크므로 기본 24시간을 둔다.
- 순방향 추적: 노출 기간에 원인 챔버를 지난 웨이퍼, 그 웨이퍼가 속한 로트. 아직 팹 아웃되지 않은
  웨이퍼(이상 구간 뒤에 팹 아웃될 웨이퍼)도 통과 시각으로 잡힌다.
- 보류 단위: 로트. 그 안에서 원인 챔버를 지난 웨이퍼는 재검사 대상으로 따로 표시한다.
- 노출 기간 추정과 추적은 정답을 받지 않는다. 정답은 score만 읽는다.
"""

from dataclasses import dataclass

import pandas as pd

DEFAULT_BUFFER_HOURS = 24


@dataclass
class Containment:
    chamber: str
    start: pd.Timestamp
    end: pd.Timestamp
    buffer_hours: float
    lots: list[str]          # 출하 보류 로트
    wafers: pd.DataFrame     # 재검사 대상: 노출 기간에 원인 챔버를 지난 웨이퍼 (wafer_id, lot_id, ts)


def exposure_window(population: pd.DataFrame, genealogy: pd.DataFrame, chamber: str,
                    buffer_hours: float = DEFAULT_BUFFER_HOURS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """이상 구간 웨이퍼들이 원인 챔버를 지난 시각의 범위에 여유 시간을 더한다."""
    passes = genealogy[(genealogy["chamber"] == chamber) & genealogy["wafer_id"].isin(population["wafer_id"])]
    if passes.empty:
        raise ValueError(f"이상 구간 웨이퍼 중 {chamber}를 지난 웨이퍼가 없음")
    pad = pd.Timedelta(hours=buffer_hours)
    return passes["ts"].min() - pad, passes["ts"].max() + pad


def trace(genealogy: pd.DataFrame, chamber: str, start: pd.Timestamp, end: pd.Timestamp,
          buffer_hours: float = 0) -> Containment:
    """[start, end]에 chamber를 지난 웨이퍼와 그 로트."""
    hit = genealogy[(genealogy["chamber"] == chamber) & genealogy["ts"].between(start, end)]
    wafers = hit[["wafer_id", "lot_id", "ts"]].sort_values("ts").reset_index(drop=True)
    return Containment(chamber, start, end, buffer_hours, sorted(wafers["lot_id"].unique()), wafers)


def contain(population: pd.DataFrame, genealogy: pd.DataFrame, chamber: str,
            buffer_hours: float = DEFAULT_BUFFER_HOURS) -> Containment:
    start, end = exposure_window(population, genealogy, chamber, buffer_hours)
    return trace(genealogy, chamber, start, end, buffer_hours)


def _pr(found: set, truth: set) -> tuple[float, float]:
    tp = len(found & truth)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(truth) if truth else 1.0
    return precision, recall


def score(c: Containment, truth_lots: list[str], truth_wafers: list[str]) -> dict:
    """정답 영향 로트, 영향 웨이퍼와 비교한 precision, recall. 정답은 여기서만 쓴다."""
    lots, true_lots = set(c.lots), set(truth_lots)
    wafers, true_wafers = set(c.wafers["wafer_id"]), set(truth_wafers)
    lot_p, lot_r = _pr(lots, true_lots)
    wafer_p, wafer_r = _pr(wafers, true_wafers)
    return {
        "buffer_hours": c.buffer_hours,
        "held_lots": len(lots),
        "lot_precision": lot_p,
        "lot_recall": lot_r,
        "missed_lots": sorted(true_lots - lots),
        "extra_lots": sorted(lots - true_lots),
        "wafer_precision": wafer_p,
        "wafer_recall": wafer_r,
    }