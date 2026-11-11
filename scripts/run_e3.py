"""E3: 시드 0~9에서 원인 분석 성능을 잰다. seed 42(개발용)는 참고로 따로 보인다.

사용법: uv run python scripts/run_e3.py            # 시드 0~9 (약 5분)
        uv run python scripts/run_e3.py --seeds 0-2

출력
  results/reports/e3_seeds.csv    시드별 결과
  results/reports/e3_summary.md   요약 (자동 생성, 해석은 실험 기록 E3)
  results/figures/e3_ranks.png    시드별 원인 챔버 순위, 방법 A 대 B
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fabtrace.analysis.experiment import evaluate_seed, summarize  # noqa: E402
from fabtrace.analysis.pipeline import simulate_and_predict  # noqa: E402
from fabtrace.data.dataset import load_wafer_data  # noqa: E402

DEV_SEED = 42


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = map(int, text.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in text.split(",")]


def plot_ranks(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["rank_a"].fillna(0), width=0.4, color="0.6", label="Method A (raw count)")
    ax.bar([i + 0.2 for i in x], df["rank_b"].fillna(0), width=0.4, color="tab:red", label="Method B (lift + test)")
    ax.set_xticks(list(x), [str(s) for s in df["seed"]])
    ax.set_xlabel("seed")
    ax.set_ylabel("rank of true cause chamber (1 = best)")
    ax.set_title("Rank of the true cause chamber by seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, summary: dict, dev: dict | None, path: Path) -> None:
    L = ["# E3 요약: 시드별 원인 분석 성능", "",
         "> `scripts/run_e3.py`가 자동 생성한다. 해석은 실험 기록 E3에 있다.", "",
         f"평가 시드: {', '.join(map(str, df['seed']))}. 분류기는 `cnn_v1`로 고정했다.", "",
         "## 요약", "", "| 항목 | 값 |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in summary.items()]
    cols = ["seed", "detected", "window", "rank_a", "rank_b", "cause_found", "false_causes",
            "rule1_days", "rule2_extra_days", "lot_recall_0h", "lot_recall_24h", "lot_precision_24h",
            "other_pattern_alarms"]
    L += ["", "## 시드별", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df[cols].iterrows():
        L.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    if dev:
        L += ["", f"참고 (개발용 seed {DEV_SEED}, 평가에서 제외): 방법 A {dev['rank_a']}위, 방법 B {dev['rank_b']}위, "
              f"24시간 로트 recall {dev['lot_recall_24h']:.3f}, precision {dev['lot_precision_24h']:.3f}"]
    L += ["", "![시드별 원인 챔버 순위](../figures/e3_ranks.png)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/s1.yaml"))
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/cnn_v1.pt"))
    parser.add_argument("--no-dev", action="store_true", help=f"참고용 seed {DEV_SEED} 실행 생략")
    args = parser.parse_args()

    data = load_wafer_data()

    def run(seed: int) -> dict:
        start = time.perf_counter()
        fab, gen, wafers, gt, preds = simulate_and_predict(args.scenario, args.fab, seed, data, args.model)
        row = evaluate_seed(seed, gen, wafers, gt, preds, fab.start_date)
        print(f"seed {seed}: A {row['rank_a']}위, B {row['rank_b']}위, 원인 {row['cause_found']}, "
              f"로트 recall 24h {row['lot_recall_24h']:.3f}  ({time.perf_counter() - start:.0f}s)", flush=True)
        return row

    df = pd.DataFrame([run(s) for s in parse_seeds(args.seeds) if s != DEV_SEED])
    dev = None if args.no_dev else run(DEV_SEED)

    reports, figures = Path("results/reports"), Path("results/figures")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "e3_seeds.csv", index=False)
    plot_ranks(df, figures / "e3_ranks.png")
    summary = summarize(df)
    write_summary(df, summary, dev, reports / "e3_summary.md")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()