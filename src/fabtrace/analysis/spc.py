"""p 관리도와 이상 구간 탐지.

- 날짜 기준: 웨이퍼의 팹 아웃 날짜(마지막 공정 시각). 불량은 웨이퍼 테스트 시점에 발견되기 때문이다.
- 입력: 분류기 예측 패턴(pred_pattern)만 쓴다. 정답 컬럼은 받지 않는다.
- 중심선: 관리한계 밖인 날을 빼고 다시 계산하는 반복 제거(SPC 1단계 방식).
- 관리한계: 날짜마다 표본 크기 n에 따라 달라진다. 정규 근사(p +- k sigma) 대신 이항분포의 정확한
  분위수로 계산한다. 하루 기대 불량 수가 1장 미만인 드문 패턴(Near-full 등)에서 정규 근사는
  불량 1장만 나와도 3sigma를 넘겨 오탐을 낸다. 꼬리 확률은 정규분포의 2sigma, 3sigma와 같게 맞춘다.
- 이상 판정: 3sigma 초과 한 점(rule1) 또는 연속 3일 중 2일이 2sigma 초과(rule2).
  rule1만으로는 완만하게 지속되는 증가를 놓칠 수 있다는 근거는 이슈 #22 댓글에 있다.
- 이상 구간: 이상인 날을 묶되, 사이에 max_gap일 이하로 빈 날은 같은 구간으로 잇는다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binom  # noqa: E402

PRED_COL = "pred_pattern"


def fab_out_days(genealogy: pd.DataFrame, start_date: str) -> pd.Series:
    """wafer_id -> 팹 아웃 날짜(1일차 = start_date 당일)."""
    last = genealogy.groupby("wafer_id")["ts"].max()
    return ((last.dt.normalize() - pd.Timestamp(start_date)).dt.days + 1).rename("day")


def daily_counts(predictions: pd.DataFrame, days: pd.Series, pattern: str) -> pd.DataFrame:
    """날짜별 웨이퍼 수 n, 예측 불량 수 defects, 비율 p."""
    if PRED_COL not in predictions.columns:
        raise ValueError(f"예측 컬럼 {PRED_COL}이 없음. 관리도는 분류기 예측만 입력으로 받는다.")
    df = predictions[["wafer_id", PRED_COL]].merge(days, left_on="wafer_id", right_index=True)
    df["is_defect"] = df[PRED_COL] == pattern
    out = df.groupby("day").agg(n=("wafer_id", "size"), defects=("is_defect", "sum"))
    out["p"] = out["defects"] / out["n"]
    return out


def p_chart(counts: pd.DataFrame, max_iter: int = 20) -> pd.DataFrame:
    """중심선을 반복 제거로 추정하고, 날짜별 관리한계와 이상 판정을 붙인다."""
    chart = counts.copy()
    keep = pd.Series(True, index=chart.index)
    for _ in range(max_iter):
        center = chart.loc[keep, "defects"].sum() / chart.loc[keep, "n"].sum()
        ucl3 = _upper_limit(chart["n"], center, TAIL_3SIGMA)
        inside = chart["p"] <= ucl3
        if inside.equals(keep):
            break
        keep = inside

    chart["center"] = center
    chart["sigma"] = np.sqrt(center * (1 - center) / chart["n"])
    chart["ucl3"] = _upper_limit(chart["n"], center, TAIL_3SIGMA)
    chart["ucl2"] = _upper_limit(chart["n"], center, TAIL_2SIGMA)
    chart["rule1"] = chart["p"] > chart["ucl3"]
    chart["rule2"] = _two_of_three_above(chart["p"] > chart["ucl2"])
    chart["alarm"] = chart["rule1"] | chart["rule2"]
    return chart


TAIL_3SIGMA = 0.00135  # 정규분포 3sigma 단측 꼬리 확률
TAIL_2SIGMA = 0.02275  # 정규분포 2sigma 단측 꼬리 확률


def _upper_limit(n: pd.Series, center: float, tail: float) -> pd.Series:
    """P(X > k) <= tail 인 가장 작은 불량 수 k를 비율로 바꾼 값. 비율이 이 값을 넘으면 이상이다."""
    k = binom.ppf(1 - tail, n.to_numpy(), center)
    return pd.Series(k / n.to_numpy(), index=n.index)


def _two_of_three_above(above: pd.Series) -> pd.Series:
    """연속 3일 창에서 2일 이상이 2sigma를 넘으면, 그 창에서 넘은 날들을 표시한다."""
    flags = above.to_numpy()
    marked = np.zeros(len(flags), dtype=bool)
    for end in range(2, len(flags)):
        window = flags[end - 2:end + 1]
        if window.sum() >= 2:
            marked[end - 2:end + 1] |= window
    return pd.Series(marked, index=above.index)


def excursions(chart: pd.DataFrame, max_gap: int = 1) -> list[tuple[int, int]]:
    """이상인 날을 묶은 (시작일, 끝일) 목록. 끝일 포함."""
    days = sorted(chart.index[chart["alarm"]])
    windows: list[tuple[int, int]] = []
    for d in days:
        if windows and d - windows[-1][1] <= max_gap + 1:
            windows[-1] = (windows[-1][0], d)
        else:
            windows.append((d, d))
    return windows


def detect(predictions: pd.DataFrame, genealogy: pd.DataFrame, pattern: str, start_date: str):
    """예측과 이력으로 관리도와 이상 구간을 한 번에 만든다."""
    chart = p_chart(daily_counts(predictions, fab_out_days(genealogy, start_date), pattern))
    return chart, excursions(chart)


def plot_p_chart(chart: pd.DataFrame, windows: list[tuple[int, int]], pattern: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (s, e) in enumerate(windows):
        ax.axvspan(s - 0.5, e + 0.5, color="tab:red", alpha=0.08, label="excursion" if i == 0 else None)
    ax.plot(chart.index, chart["p"], color="0.2", marker="o", ms=3, lw=1, label=f"daily {pattern} rate")
    ax.plot(chart.index, chart["center"], color="tab:green", lw=1, label="center")
    ax.step(chart.index, chart["ucl2"], where="mid", color="tab:orange", lw=0.8, ls="--", label="2 sigma")
    ax.step(chart.index, chart["ucl3"], where="mid", color="tab:red", lw=1, label="UCL (3 sigma)")
    r1, r2 = chart[chart["rule1"]], chart[chart["rule2"] & ~chart["rule1"]]
    ax.scatter(r1.index, r1["p"], color="tab:red", s=40, zorder=3, label="rule1: beyond 3 sigma")
    ax.scatter(r2.index, r2["p"], facecolor="none", edgecolor="tab:red", s=60, zorder=3,
               label="rule2: 2 of 3 beyond 2 sigma")
    ax.set_xlabel("day (fab out)")
    ax.set_ylabel(f"predicted {pattern} rate")
    ax.set_title(f"p-chart: predicted {pattern} rate by fab-out day")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)