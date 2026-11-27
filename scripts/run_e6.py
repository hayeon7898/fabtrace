"""E6: 고정 파이프라인의 시나리오별 기준선 (#49).

사용법: uv run python scripts/run_e6.py                     # S1, S2 강도 목록, S3, S4, S5 x 개발 시드 200~209
        uv run python scripts/run_e6.py --scenarios S1,S3   # 일부만
        uv run python scripts/run_e6.py --scenarios S1@sweep # S1 난이도 곡선 (있으면 좋음)
        uv run python scripts/run_e6.py --rescore-e4        # 참고: E4 시드 100~109를 새 채점으로 다시 채점
        uv run python scripts/run_e6.py --summary-only      # 이미 있는 csv로 요약과 그림만 다시 만든다

- 방법은 exposure+chamber(현재 최선의 고정 파이프라인)다. 채점은 scoring.score_general이다.
- S2는 강도 목록(scenario.INTENSITY_SWEEP)마다 같은 정의를 돌린다. S2@0.2가 s2.yaml과 같다.
- 한 회가 끝날 때마다 csv를 저장한다. 중간에 끊겨도 다시 실행하면 남은 회만 돈다.
- 평가 시드(300~319)는 거부한다. 판정 기준은 experiment.judge_e6에 있다.
- 시나리오 x 시드 한 회는 예측을 포함해 약 30초다. 기본 실행은 90회, 약 45분이다.

출력
  results/reports/e6_seeds.csv         시나리오 x 시드별 결과
  results/reports/e6_summary.md        가설 판정, 시나리오별 표 (자동 생성, 해석은 실험 기록 E6)
  results/figures/e6_s2_sweep.png      S2 강도별 원인 특정률과 지목한 챔버 비율
  results/reports/e6_rescore_e4.csv    --rescore-e4: E4 시드의 첫 구간 기준과 합친 기준 lot recall
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fabtrace.analysis.experiment import (DEV_SEEDS, E6_MAX_MISSING_SEEDS, E6_MIN_BOTH_FOUND,  # noqa: E402
                                          E6_MIN_FAIL_SEEDS, E6_MIN_S5_CLEAN, check_not_eval,
                                          evaluate_scenario, judge_e6)
from fabtrace.sim.config import load_fab_config  # noqa: E402
from fabtrace.sim.scenario import INTENSITY_SWEEP, load_scenario, with_intensity  # noqa: E402

SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_SCENARIOS = "S1,S2@sweep,S3,S4,S5"
E4_SEEDS = tuple(range(100, 110))
# 기본 팔레트의 범주형 1, 2번
COLORS = {"found": "#2a78d6", "fraction": "#eb6834"}


def expand_scenarios(text: str, fab) -> list:
    """'S1,S2@sweep,S2@0.3' 같은 목록을 (이름, Scenario) 목록으로."""
    out = []
    for name in text.split(","):
        base, _, level = name.strip().partition("@")
        sc = load_scenario(SCENARIO_DIR / f"{base.lower()}.yaml", fab)
        if level == "sweep":
            out += [(f"{base}@{i:g}", with_intensity(sc, i)) for i in INTENSITY_SWEEP]
        elif level:
            out.append((f"{base}@{float(level):g}", with_intensity(sc, float(level))))
        else:
            out.append((base, sc))
    return out


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = map(int, text.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in text.split(",")]


def run(scenarios: list, seeds: list[int], simulate, csv_path: Path) -> pd.DataFrame:
    """simulate(scenario, seed) -> (fab, genealogy, wafers, gt, predictions). 이미 있는 회는 건너뛴다."""
    check_not_eval(seeds)
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=["scenario", "seed"])
    have = set(zip(done["scenario"], done["seed"]))
    for name, sc in scenarios:
        for seed in seeds:
            if (name, seed) in have:
                continue
            t = time.perf_counter()
            fab, genealogy, wafers, gt, preds = simulate(sc, seed)
            row = evaluate_scenario(name, seed, genealogy, wafers, gt, preds, fab.start_date)
            # 원인 수에 따라 열(c0_, c1_)이 달라지므로 이어 붙이지 않고 매번 전체를 다시 쓴다
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(csv_path, index=False)
            print(f"{name} seed {seed}: 원인 {row['causes_found']}/{row['n_causes']}, "
                  f"진짜가 아닌 원인 {row['n_wrong_causes']}, 보류 {row['held_lots_total']}  "
                  f"({time.perf_counter() - t:.0f}s)", flush=True)
    return pd.read_csv(csv_path)


def plot_sweep(df: pd.DataFrame, path: Path) -> None:
    g = df[df["scenario"].str.startswith("S2@")].copy()
    if g.empty:
        return
    g["intensity"] = g["scenario"].str.split("@").str[1].astype(float)
    s = g.groupby("intensity").agg(found=("c0_found", "mean"), fraction=("c0_chamber_fraction", "mean"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(s.index, s["found"], marker="o", lw=2, color=COLORS["found"], label="cause found (all 3 chambers)")
    ax.plot(s.index, s["fraction"], marker="o", lw=2, color=COLORS["fraction"], label="share of chambers named")
    ax.set_xlabel("fault intensity (S2, tool ETCH-02)")
    ax.set_ylabel("rate over seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("S2 difficulty curve, fixed pipeline", pad=28)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _v(x) -> str:
    if isinstance(x, float):
        return "-" if pd.isna(x) else f"{x:.2f}"
    return str(x)


def write_summary(df: pd.DataFrame, path: Path, figure: Path) -> None:
    L = ["# E6 요약: 고정 파이프라인의 시나리오별 기준선", "",
         "> `scripts/run_e6.py`가 자동 생성한다. 해석은 실험 기록 E6에 있다.", "",
         f"방법 `exposure+chamber`, 시드 {', '.join(map(str, sorted(df['seed'].unique())))}. 채점은 `scoring.score_general`.", "",
         "## 가설 판정", "",
         f"기준(실행 전에 정함): 실패는 10개 중 {E6_MIN_FAIL_SEEDS}개 이상, S1 로트를 놓친 시드 {E6_MAX_MISSING_SEEDS}개 이하, "
         f"S3 두 원인 모두 찾음 {E6_MIN_BOTH_FOUND}개 이상, S5 원인 없음 {E6_MIN_S5_CLEAN}개 이상.", "",
         "| 가설 | 판정 | 근거 |", "| --- | --- | --- |"]
    L += [f"| {j['hypothesis']} | {j['verdict']} | {j['evidence']} |" for j in judge_e6(df)]

    agg = df.groupby("scenario", sort=False).agg(
        seeds=("seed", "size"), causes_found=("causes_found", "sum"), n_causes=("n_causes", "sum"),
        cause_precision=("cause_precision", "mean"), wrong_per_seed=("n_wrong_causes", "mean"),
        lot_recall_total=("lot_recall_total", "mean"), extra_lots_total=("extra_lots_total", "mean"),
        held_lots_total=("held_lots_total", "mean"))
    L += ["", "## 시나리오별", "",
          "| 시나리오 | 시드 | 찾은 원인 / 정답 원인 | 원인 precision | 시드당 진짜가 아닌 원인 | lot recall (전체) "
          "| 추가 보류 (전체) | 보류 로트 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in agg.iterrows():
        L.append(f"| {name} | {int(r['seeds'])} | {int(r['causes_found'])}/{int(r['n_causes'])} | {_v(r['cause_precision'])} "
                 f"| {_v(r['wrong_per_seed'])} | {_v(r['lot_recall_total'])} | {_v(r['extra_lots_total'])} "
                 f"| {_v(r['held_lots_total'])} |")
    if figure.exists():
        L += ["", f"![S2 난이도 곡선](../figures/{figure.name})"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def rescore_e4(args, simulate_path, out: Path) -> None:
    """E4 시드를 baseline, exposure+chamber로 다시 돌려 첫 구간 기준과 합친 기준 lot recall을 비교한다."""
    rows = []
    for seed in E4_SEEDS:
        fab, genealogy, wafers, gt, preds = simulate_path(SCENARIO_DIR / "s1.yaml", seed)
        for method in ("baseline", "exposure+chamber"):
            r = evaluate_scenario("S1", seed, genealogy, wafers, gt, preds, fab.start_date, method)
            rows.append({k: r[k] for k in ("seed", "method", "c0_detected", "c0_found", "c0_found_first",
                                           "c0_lot_recall_first", "c0_lot_recall", "n_wrong_causes")})
        print(f"E4 seed {seed} 다시 채점", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby("method")[["c0_lot_recall_first", "c0_lot_recall"]].mean().round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--seeds", default=f"{DEV_SEEDS[0]}-{DEV_SEEDS[-1]}")
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/cnn_v1.pt"))
    parser.add_argument("--rescore-e4", action="store_true", help="참고: E4 시드 100~109를 새 채점으로 다시 채점")
    parser.add_argument("--summary-only", action="store_true", help="이미 있는 e6_seeds.csv로 요약과 그림만 다시 만든다")
    args = parser.parse_args()

    reports, figures = Path("results/reports"), Path("results/figures")
    if args.summary_only:
        df = pd.read_csv(reports / "e6_seeds.csv")
        plot_sweep(df, figures / "e6_s2_sweep.png")
        write_summary(df, reports / "e6_summary.md", figures / "e6_s2_sweep.png")
        print(f"  summary: {reports / 'e6_summary.md'}")
        return

    seeds = parse_seeds(args.seeds)
    check_not_eval(seeds)  # 데이터를 읽기 전에 거부한다

    from fabtrace.analysis.pipeline import simulate_and_predict
    from fabtrace.data.dataset import load_wafer_data

    data = load_wafer_data()

    def simulate(sc, seed):
        return simulate_and_predict(sc, args.fab, seed, data, args.model)

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if args.rescore_e4:
        rescore_e4(args, simulate, reports / "e6_rescore_e4.csv")
        return

    fab = load_fab_config(args.fab)
    df = run(expand_scenarios(args.scenarios, fab), seeds, simulate, reports / "e6_seeds.csv")
    plot_sweep(df, figures / "e6_s2_sweep.png")
    write_summary(df, reports / "e6_summary.md", figures / "e6_s2_sweep.png")
    for j in judge_e6(df):
        print(f"  가설 {j['hypothesis']}: {j['verdict']} ({j['evidence']})")
    print(f"  summary: {reports / 'e6_summary.md'}")


if __name__ == "__main__":
    main()
