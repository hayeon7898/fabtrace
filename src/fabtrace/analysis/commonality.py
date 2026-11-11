"""공통성 분석: 이상 구간의 불량 웨이퍼가 공통으로 거친 챔버를 찾는다.

- 대상: 이상 구간(팹 아웃 날짜 기준)에 팹 아웃된 웨이퍼. 불량 여부는 분류기 예측으로 정한다.
- 비교: 구간 안에서 그 챔버를 거친 웨이퍼와 같은 단계의 다른 챔버를 거친 웨이퍼.
- 방법 A: 불량 웨이퍼가 거친 횟수로 순위. 처리량이 많은 챔버, 특히 모든 웨이퍼가 거치는 병목 설비가 위로 온다.
- 방법 B: lift(거친 웨이퍼 불량률 / 구간 전체 불량률), 단측 피셔 정확 검정, Benjamini-Hochberg 보정.
  q-value 오름차순, 같으면 lift 내림차순으로 정렬한다.
  모든 웨이퍼가 거친 챔버는 비교 대상이 없어 p = 1로 두므로 원인 후보에서 자연히 빠진다.
"""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

from fabtrace.analysis.spc import PRED_COL, fab_out_days

ALPHA = 0.05


def window_population(predictions: pd.DataFrame, genealogy: pd.DataFrame, window: tuple[int, int],
                      start_date: str, pattern: str) -> pd.DataFrame:
    """이상 구간에 팹 아웃된 웨이퍼와 예측 불량 여부."""
    if PRED_COL not in predictions.columns:
        raise ValueError(f"예측 컬럼 {PRED_COL}이 없음. 공통성 분석은 분류기 예측만 입력으로 받는다.")
    days = fab_out_days(genealogy, start_date)
    pop = predictions[["wafer_id", PRED_COL]].merge(days, left_on="wafer_id", right_index=True)
    pop = pop[pop["day"].between(*window)]
    return pop.assign(is_defect=pop[PRED_COL] == pattern)[["wafer_id", "day", "is_defect"]]


def chamber_table(population: pd.DataFrame, genealogy: pd.DataFrame) -> pd.DataFrame:
    """챔버별 2x2 집계: 거친 웨이퍼(passed)와 같은 단계의 나머지(not_passed)의 불량 수."""
    hist = genealogy[["wafer_id", "step", "tool", "chamber"]].merge(population, on="wafer_id")
    per = hist.groupby(["step", "tool", "chamber"]).agg(passed=("wafer_id", "size"),
                                                         defect_passed=("is_defect", "sum")).reset_index()
    step_tot = hist.groupby("step").agg(step_n=("wafer_id", "size"), step_defects=("is_defect", "sum"))
    per = per.merge(step_tot, left_on="step", right_index=True)
    per["not_passed"] = per["step_n"] - per["passed"]
    per["defect_not_passed"] = per["step_defects"] - per["defect_passed"]
    overall = population["is_defect"].mean()
    per["defect_rate"] = per["defect_passed"] / per["passed"]
    per["lift"] = per["defect_rate"] / overall if overall > 0 else np.nan
    return per.drop(columns=["step_n", "step_defects"])


def method_a(table: pd.DataFrame) -> pd.DataFrame:
    out = table.sort_values(["defect_passed", "passed"], ascending=[False, True]).reset_index(drop=True)
    return out.assign(rank=range(1, len(out) + 1))


def method_b(table: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    def one_sided_p(r) -> float:
        if r["not_passed"] == 0 or r["passed"] == 0:
            return 1.0  # 비교 대상이 없으면 원인으로 판단할 근거가 없다
        tab = [[r["defect_passed"], r["passed"] - r["defect_passed"]],
               [r["defect_not_passed"], r["not_passed"] - r["defect_not_passed"]]]
        return float(fisher_exact(tab, alternative="greater").pvalue)

    out = table.copy()
    out["p"] = out.apply(one_sided_p, axis=1)
    out["q"] = false_discovery_control(out["p"].to_numpy(), method="bh")
    out["significant"] = out["q"] < alpha
    out = out.sort_values(["q", "lift", "defect_passed"], ascending=[True, False, False]).reset_index(drop=True)
    return out.assign(rank=range(1, len(out) + 1))


def analyze(predictions: pd.DataFrame, genealogy: pd.DataFrame, window: tuple[int, int],
            start_date: str, pattern: str) -> dict[str, pd.DataFrame]:
    pop = window_population(predictions, genealogy, window, start_date, pattern)
    table = chamber_table(pop, genealogy)
    return {"population": pop, "A": method_a(table), "B": method_b(table)}


def rank_of(ranking: pd.DataFrame, chamber: str) -> int:
    return int(ranking.loc[ranking["chamber"] == chamber, "rank"].iloc[0])