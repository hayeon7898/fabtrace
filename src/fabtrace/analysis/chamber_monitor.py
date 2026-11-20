"""챔버 단위 감시: 패턴마다, 챔버마다 통과 날짜별 예측 불량률을 감시한다.

공장 전체 관리도(spc)는 모든 웨이퍼를 팹 아웃 날짜로 모아서 보므로, 챔버 하나의 이상이 묽어진다
(E3에서 2/10 시드를 놓침). 챔버를 따로 보면 원인 챔버 안 불량률은 약 40%, 평상시는 약 5%다.
공장 전체 감시는 그대로 두고 함께 쓴다. 이 모듈은 알림만 내고 원인을 확정하지 않는다.
알림은 공통성 분석으로 확인한다(#36).

- 날짜 기준: 웨이퍼가 그 챔버를 지난 날짜(통과 날짜). 입력은 분류기 예측뿐이다.
- 사후 집계: 불량은 팹 아웃 뒤 웨이퍼 테스트와 분류를 거쳐야 알 수 있다. 통과 날짜 기준 집계는
  그 날짜에 챔버를 지난 웨이퍼가 모두 팹 아웃되어 판정이 나온 뒤에야 완성된다. 실시간 감시가 아니다.
- 중심선: 패턴마다 공장 전체 관리도의 중심선(반복 제거)을 모든 챔버에 쓴다. 챔버 자신의 중심선은
  드문 패턴(Near-full 등)에서 40일 동안 불량이 0장이라 0이 되기 쉽고, 그러면 1장만 나와도 알린다.
  가상 팹에서는 원인 챔버 외의 모든 챔버가 같은 비율을 따르므로 이 가정이 맞다. 실제 팹에서는
  챔버마다 평상시 비율이 다를 수 있다.
- 판정: 관리한계 초과 한 점(rule1)만 쓴다. 동시에 감시하는 점(챔버 x 패턴 x 웨이퍼가 있는 날)이
  약 8,000개라, 공장 전체 관리도처럼 3sigma와 rule2를 쓰면 한 번 실행에 오탐이 수십 건 나온다.
- 꼬리 확률: 오탐 예산(실행당 기대 오탐 수) / 감시 점 수. 점마다 오탐 확률이 꼬리 확률 이하이므로
  실행당 기대 오탐 점 수는 예산 이하이고, 오탐 구간 수는 점 수보다 많을 수 없다. 기본 예산은 스펙
  기준 1건의 절반인 0.5로, 중심선 추정 오차에 여유를 둔다. 이항분포가 이산적이라 실제 오탐률은
  꼬리 확률보다 낮다.
- 알림 구간: 알린 날을 하루 간격까지 잇는다. 그 챔버를 지난 웨이퍼가 없는 날은 간격으로 세지 않는다
  (exposure.chamber_runs). 오탐은 구간 단위로 센다.
"""

from dataclasses import dataclass, field

import pandas as pd

from fabtrace.analysis.exposure import chamber_runs
from fabtrace.analysis.spc import PRED_COL, detect, p_chart

DEFAULT_BUDGET = 0.5
ALARM_COLUMNS = ["pattern", "step", "chamber", "start_day", "end_day", "alarm_days", "max_p", "center"]


@dataclass
class ChamberMonitor:
    budget: float
    n_points: int
    tail: float
    centers: dict[str, float]
    alarms: pd.DataFrame                    # ALARM_COLUMNS, 알림 구간 하나가 한 행
    charts: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)  # (pattern, chamber) -> 관리도


def pass_day_counts(predictions: pd.DataFrame, genealogy: pd.DataFrame, start_date: str,
                    patterns: list[str]) -> pd.DataFrame:
    """(step, chamber, 통과 날짜)별 웨이퍼 수 n과 패턴별 예측 불량 수."""
    if PRED_COL not in predictions.columns:
        raise ValueError(f"예측 컬럼 {PRED_COL}이 없음. 챔버 감시는 분류기 예측만 입력으로 받는다.")
    df = genealogy[["wafer_id", "step", "chamber", "ts"]].merge(predictions[["wafer_id", PRED_COL]], on="wafer_id")
    df["day"] = (df["ts"].dt.normalize() - pd.Timestamp(start_date)).dt.days + 1
    for p in patterns:
        df[p] = df[PRED_COL] == p
    agg = {"n": ("wafer_id", "size")} | {p: (p, "sum") for p in patterns}
    return df.groupby(["step", "chamber", "day"]).agg(**agg).reset_index()


def fab_centers(predictions: pd.DataFrame, genealogy: pd.DataFrame, start_date: str,
                patterns: list[str]) -> dict[str, float]:
    """패턴마다 공장 전체 관리도의 중심선."""
    return {p: float(detect(predictions, genealogy, p, start_date)[0]["center"].iloc[0]) for p in patterns}


def monitor(predictions: pd.DataFrame, genealogy: pd.DataFrame, start_date: str, patterns: list[str],
            centers: dict[str, float] | None = None, budget: float = DEFAULT_BUDGET) -> ChamberMonitor:
    """패턴 x 챔버마다 통과 날짜별 관리도를 그리고 알림 구간을 모은다."""
    preds = predictions[["wafer_id", PRED_COL]]  # 예측 외의 컬럼은 들여오지 않는다
    if centers is None:
        centers = fab_centers(preds, genealogy, start_date, patterns)
    counts = pass_day_counts(preds, genealogy, start_date, patterns)
    n_points = len(counts) * len(patterns)
    tail = budget / n_points

    rows, charts = [], {}
    for (step, chamber), g in counts.groupby(["step", "chamber"], sort=True):
        g = g.set_index("day").sort_index()
        for p in patterns:
            c = g[["n"]].assign(defects=g[p], p=g[p] / g["n"])
            chart = p_chart(c, center=centers[p], tail3=tail)
            charts[(p, chamber)] = chart
            for s, e in chamber_runs(chart.assign(alarm=chart["rule1"])):
                span = chart.loc[s:e]
                rows.append({"pattern": p, "step": step, "chamber": chamber, "start_day": s, "end_day": e,
                             "alarm_days": int(span["rule1"].sum()), "max_p": float(span["p"].max()),
                             "center": centers[p]})
    alarms = pd.DataFrame(rows, columns=ALARM_COLUMNS)
    alarms = alarms.sort_values(["pattern", "start_day", "chamber"]).reset_index(drop=True)
    return ChamberMonitor(budget, n_points, tail, centers, alarms, charts)
