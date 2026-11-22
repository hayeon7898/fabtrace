"""E4: 평가 시드 100~109에서 기존 방법과 개선 방법을 같은 조건으로 비교한다 (#37).

사용법: uv run python scripts/run_e4.py              # 시드 100~109 (분류기와 데이터 필요)
        uv run python scripts/run_e4.py --seeds 100-101

- 시드마다 가상 팹과 cnn_v1 예측을 한 번만 만들고 baseline, exposure(A), exposure+chamber(A+B)를 모두 적용한다.
- 가설은 baseline 대 exposure+chamber, 여유 24시간으로 판정한다(experiment.judge_e4). exposure는 A의 효과를
  따로 보기 위한 참고다. 여유 24시간은 E5에서 유지하기로 했다.
- 시드 0~9는 E3 결과를 보고 개선 방향을 정했으므로, seed 42는 개발에 썼으므로 쓰지 않는다.

출력
  results/reports/e4_seeds.csv    시드 x 방법별 결과
  results/reports/e4_summary.md   방법별 요약, 시드별 표, 가설 판정 (자동 생성, 해석은 실험 기록 E4)
  results/figures/e4_recall.png   시드별 로트 recall (여유 24시간), 방법별
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fabtrace.analysis.experiment import (E4_BUFFER, E4_MAX_EXTRA_LOTS, E4_MAX_MISSING_SEEDS,  # noqa: E402
                                          E4_METHODS, evaluate_methods, judge_e4, summarize_e4)
from fabtrace.analysis.pipeline import simulate_and_predict  # noqa: E402
from fabtrace.data.dataset import load_wafer_data  # noqa: E402

# 기본 팔레트의 범주형 1~3번(파랑, 주황, 청록). 방법마다 색을 고정한다.
COLORS = {"baseline": "#2a78d6", "exposure": "#eb6834", "exposure+chamber": "#1baf7a"}
LABELS = {"baseline": "baseline (E3)", "exposure": "A: exposure", "exposure+chamber": "A+B: exposure+chamber"}


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = map(int, text.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in text.split(",")]


def plot_recall(df: pd.DataFrame, path: Path) -> None:
    seeds = sorted(df["seed"].unique())
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(E4_METHODS):
        g = df[df["method"] == m].set_index("seed").loc[seeds]
        x = [j + (i - 1) * (width + 0.02) for j in range(len(seeds))]
        ax.bar(x, g[f"lot_recall_{E4_BUFFER}h"], width=width, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("seed")
    ax.set_ylabel(f"lot recall ({E4_BUFFER}h margin)")
    ax.set_title(f"Lot recall by seed and method ({E4_BUFFER}h margin)", pad=28)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _v(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    b = E4_BUFFER
    summary = summarize_e4(df)
    L = ["# E4 요약: 챔버 단위 개선 효과", "",
         "> `scripts/run_e4.py`가 자동 생성한다. 해석은 실험 기록 E4에 있다.", "",
         f"평가 시드: {', '.join(map(str, sorted(df['seed'].unique())))}. 분류기는 `cnn_v1`, 여유 {b}시간.", "",
         "## 가설 판정", "",
         f"기준(실행 전에 정함): 가설 2는 로트를 놓친 시드 {E4_MAX_MISSING_SEEDS}개 이하, "
         f"가설 3은 추가 보류 정상 로트 평균 {E4_MAX_EXTRA_LOTS}개 이하. baseline 대 exposure+chamber.", "",
         "| 가설 | 판정 | 근거 |", "| --- | --- | --- |"]
    L += [f"| {j['hypothesis']} | {j['verdict']} | {j['evidence']} |" for j in judge_e4(df)]

    cols = list(summary.columns)
    L += ["", "## 방법별 요약", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    L += ["| " + " | ".join(_v(v) for v in r) + " |" for r in summary.itertuples(index=False)]

    per = ["seed", "method", "detected", "window", "rank_b", "cause_found", "wrong_causes_all",
           f"lot_recall_{b}h", f"lot_precision_{b}h", f"extra_lots_{b}h", "chamber_alarms", "alarms_confirmed"]
    L += ["", "## 시드별", "", "| " + " | ".join(per) + " |", "| " + " | ".join("---" for _ in per) + " |"]
    order = df.assign(_m=df["method"].map({m: i for i, m in enumerate(E4_METHODS)})).sort_values(["seed", "_m"])
    for r in order[per].itertuples(index=False):
        L.append("| " + " | ".join(_v(v) for v in r) + " |")
    L += ["", "![시드별 로트 recall](../figures/e4_recall.png)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100-109")
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/s1.yaml"))
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/cnn_v1.pt"))
    args = parser.parse_args()

    data = load_wafer_data()
    rows = []
    for seed in parse_seeds(args.seeds):
        t = time.time()
        fab, genealogy, wafers, gt, preds = simulate_and_predict(args.scenario, args.fab, seed, data, args.model)
        out = evaluate_methods(seed, genealogy, wafers, gt, preds, fab.start_date)
        rows += out
        brief = ", ".join(f"{r['method']} 감지 {r['detected']} recall {r[f'lot_recall_{E4_BUFFER}h']:.3f}" for r in out)
        print(f"seed {seed}: {brief}  ({time.time() - t:.0f}s)")

    df = pd.DataFrame(rows)
    reports, figures = Path("results/reports"), Path("results/figures")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "e4_seeds.csv", index=False)
    plot_recall(df, figures / "e4_recall.png")
    write_summary(df, reports / "e4_summary.md")
    for j in judge_e4(df):
        print(f"  가설 {j['hypothesis']}: {j['verdict']} ({j['evidence']})")
    print(f"  summary: {reports / 'e4_summary.md'}")


if __name__ == "__main__":
    main()
