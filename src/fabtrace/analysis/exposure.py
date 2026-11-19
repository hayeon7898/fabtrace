"""원인 챔버 기준 노출 기간 재추정.

기존 방법(containment.exposure_window)은 관리도 이상 구간에 팹 아웃된 웨이퍼들이 원인 챔버를 지난
시각으로 노출 기간을 정한다. 이상 구간은 관리도 정점 며칠이라, 노출 기간이 실제보다 짧게 잘린다
(E3에서 9/10 시드가 로트를 놓침). 원인 챔버가 정해진 뒤에는 그 챔버를 지난 웨이퍼만 보면 신호가
뚜렷하므로(이상 기간 약 40%, 평상시 약 5%), 그 챔버의 관리도로 노출 기간을 다시 잡는다.

- 날짜 기준: 원인 챔버를 지난 날짜(통과 날짜). 팹 아웃 날짜가 아니다.
- 집계: 그 챔버를 지난 웨이퍼만, 전체 기간에서 날짜별 예측 불량률을 센다. 입력은 분류기 예측뿐이다.
- 관리도: spc.p_chart를 그대로 쓴다. 중심선은 그 챔버 자신의 전체 기간에서 반복 제거로 구하고,
  관리한계는 이항분포 분위수, 이상 판정은 rule1 또는 rule2로 공장 전체 관리도와 같다.
- 이상 구간: 이상인 날을 묶되 하루 간격까지 잇는다. 그 챔버를 지난 웨이퍼가 없는 날은 판단 근거가
  없으므로 간격으로 세지 않는다. 로트가 설비를 통째로 고르므로 이런 날이 가끔 생긴다.
- 기준 구간: 기존 방법의 노출 기간(여유 시간 전). 기준 구간과 겹치는 챔버 이상 구간을 고른다.
- 노출 기간: 고른 챔버 이상 구간의 날들에 실제로 챔버를 지난 시각의 처음과 끝, 그리고 기준 구간의
  합집합. 기존 방법보다 짧아지지 않도록 합친다. 불량 로트를 놓치는 비용이 더 크기 때문이다.
  겹치는 챔버 이상 구간이 없으면 기준 구간을 그대로 쓴다.
- 여유 시간: 기존처럼 노출 기간 앞뒤에 더한다.
- 정답은 받지 않는다. 채점은 containment.score가 한다.
"""

from dataclasses import dataclass

import pandas as pd

from fabtrace.analysis.containment import DEFAULT_BUFFER_HOURS, Containment, exposure_window, trace
from fabtrace.analysis.spc import PRED_COL, excursions, p_chart

MAX_GAP_DAYS = 1


@dataclass
class ChamberExposure:
    chamber: str
    pattern: str
    chart: pd.DataFrame                       # 통과 날짜별 관리도 (spc.p_chart 형식)
    runs: list[tuple[int, int]]               # 챔버 관리도의 이상 구간 전체 (통과 날짜, 끝일 포함)
    anchor: tuple[pd.Timestamp, pd.Timestamp]  # 기존 방법의 노출 기간, 여유 시간 전
    selected: tuple[int, int] | None          # 기준 구간과 겹쳐 고른 이상 구간. 없으면 None
    start: pd.Timestamp                       # 노출 기간, 여유 시간 전
    end: pd.Timestamp

    @property
    def method(self) -> str:
        return "chamber" if self.selected is not None else "anchor"


def pass_days(genealogy: pd.DataFrame, chamber: str, start_date: str) -> pd.DataFrame:
    """chamber를 지난 웨이퍼의 통과 시각과 통과 날짜(1일차 = start_date 당일)."""
    passes = genealogy.loc[genealogy["chamber"] == chamber, ["wafer_id", "ts"]]
    day = (passes["ts"].dt.normalize() - pd.Timestamp(start_date)).dt.days + 1
    return passes.assign(day=day).reset_index(drop=True)


def chamber_daily_counts(predictions: pd.DataFrame, genealogy: pd.DataFrame, chamber: str,
                         pattern: str, start_date: str) -> pd.DataFrame:
    """chamber를 지난 웨이퍼만의 통과 날짜별 웨이퍼 수 n, 예측 불량 수 defects, 비율 p."""
    if PRED_COL not in predictions.columns:
        raise ValueError(f"예측 컬럼 {PRED_COL}이 없음. 노출 기간 추정은 분류기 예측만 입력으로 받는다.")
    df = pass_days(genealogy, chamber, start_date).merge(predictions[["wafer_id", PRED_COL]], on="wafer_id")
    df["is_defect"] = df[PRED_COL] == pattern
    out = df.groupby("day").agg(n=("wafer_id", "size"), defects=("is_defect", "sum"))
    out["p"] = out["defects"] / out["n"]
    return out


def chamber_runs(chart: pd.DataFrame, max_gap: int = MAX_GAP_DAYS) -> list[tuple[int, int]]:
    """이상인 날을 묶은 (시작일, 끝일) 목록. 웨이퍼가 없어 관리도에 없는 날은 간격으로 세지 않는다."""
    positional = chart.reset_index(drop=True)
    return [(int(chart.index[s]), int(chart.index[e])) for s, e in excursions(positional, max_gap)]


def select_runs(runs: list[tuple[int, int]], anchor_days: tuple[int, int]) -> tuple[int, int] | None:
    """기준 구간과 겹치는 이상 구간들을 하나로 합친 (시작일, 끝일). 겹치는 구간이 없으면 None."""
    lo, hi = anchor_days
    hit = [(s, e) for s, e in runs if s <= hi and e >= lo]
    if not hit:
        return None
    return min(s for s, _ in hit), max(e for _, e in hit)


def estimate_exposure(population: pd.DataFrame, predictions: pd.DataFrame, genealogy: pd.DataFrame,
                      chamber: str, pattern: str, start_date: str) -> ChamberExposure:
    """원인 챔버의 통과 날짜별 관리도로 노출 기간을 다시 잡는다. 여유 시간은 더하지 않는다."""
    anchor = exposure_window(population, genealogy, chamber, buffer_hours=0)
    passes = pass_days(genealogy, chamber, start_date)
    anchor_days = tuple(int(passes.loc[passes["ts"] == t, "day"].iloc[0]) for t in anchor)

    chart = p_chart(chamber_daily_counts(predictions, genealogy, chamber, pattern, start_date))
    runs = chamber_runs(chart)
    selected = select_runs(runs, anchor_days)

    start, end = anchor
    if selected is not None:
        in_run = passes.loc[passes["day"].between(*selected), "ts"]
        start, end = min(start, in_run.min()), max(end, in_run.max())
    return ChamberExposure(chamber, pattern, chart, runs, anchor, selected, start, end)


def contain_by_chamber(population: pd.DataFrame, predictions: pd.DataFrame, genealogy: pd.DataFrame,
                       chamber: str, pattern: str, start_date: str,
                       buffer_hours: float = DEFAULT_BUFFER_HOURS) -> tuple[Containment, ChamberExposure]:
    """estimate_exposure로 잡은 노출 기간에 여유 시간을 더해 순방향 추적한다."""
    ex = estimate_exposure(population, predictions, genealogy, chamber, pattern, start_date)
    pad = pd.Timedelta(hours=buffer_hours)
    return trace(genealogy, chamber, ex.start - pad, ex.end + pad, buffer_hours), ex
