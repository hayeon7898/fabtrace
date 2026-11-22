"""E5: 개선 방법의 기본 격리 여유 시간을 E4 전에 seed 42만으로 다시 정한다 (#41).

사용법: uv run python scripts/run_e5.py               # 실제 예측 + 합성 예측 50개 (분류기와 데이터 필요)
        uv run python scripts/run_e5.py --skip-real   # 합성 예측만 (분류기와 데이터 불필요, 결론은 내지 않음)

- 방법: exposure+chamber. seed 42 실제 예측은 공장 전체 감시가 이상을 찾으므로 exposure와 결과가 같다.
  합성 예측은 공장 전체 감시가 놓치는 경우가 있어, 그때 여유 시간과 무관하게 recall이 0이 되지 않도록
  챔버 감시까지 켠 모드로 센다.
- 실제 예측: seed 42 가상 팹과 cnn_v1 예측.
- 합성 예측: seed 42 이력은 그대로 두고 맵 배정만 시드 0~49로 바꾼다. 평상시 패턴은 WM-811K 클래스 비율,
  영향 웨이퍼는 S1 intensity(0.4) 확률로 Edge-Ring이다. 예측은 배정된 패턴 그대로다(분류 오차 없음).
- 판단 기준은 experiment.choose_buffer에 있다. 결과를 보기 전에 #41에 적은 기준이다.
- 평가 시드(100~109)는 쓰지 않는다.

출력
  results/reports/e5_real.csv        실제 예측, 여유 시간별 결과
  results/reports/e5_synthetic.csv   합성 예측, 시드별 x 여유 시간별 결과
  results/reports/e5_summary.md      여유 시간별 판정 표와 선택한 값 (자동 생성, 해석은 실험 기록 E5)
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from fabtrace.analysis.experiment import (CURRENT_BUFFER, MIN_SYNTH_RECALL_MEAN, MIN_SYNTH_RECALL_MIN,
                                          REVIEW_BUFFERS, WM811K_CLASS_COUNTS, choose_buffer, evaluate_seed)
from fabtrace.sim.config import load_fab_config
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth
from fabtrace.sim.scenario import load_scenario

DEV_SEED = 42
METHOD = "exposure+chamber"


def synthetic_pool() -> pd.DataFrame:
    labels = [c for c, n in WM811K_CLASS_COUNTS.items() for _ in range(n)]
    return pd.DataFrame({"label": labels}, index=range(len(labels)))


def run_synthetic(fab, sc, draws: int) -> pd.DataFrame:
    genealogy = generate_genealogy(fab, DEV_SEED)
    pool = synthetic_pool()
    rows = []
    for draw in range(draws):
        wafers = build_wafers(genealogy, sc, fab, pool, draw)
        preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
        gt = ground_truth(sc, fab, wafers, draw)
        rows.append(evaluate_seed(draw, genealogy, wafers, gt, preds, fab.start_date, METHOD, REVIEW_BUFFERS))
    return pd.DataFrame(rows).rename(columns={"seed": "draw"})


def run_real(args) -> dict:
    from fabtrace.analysis.pipeline import simulate_and_predict
    from fabtrace.data.dataset import load_wafer_data

    _, genealogy, wafers, gt, preds = simulate_and_predict(args.scenario, args.fab, DEV_SEED,
                                                           load_wafer_data(), args.model)
    fab = load_fab_config(args.fab)
    return evaluate_seed(DEV_SEED, genealogy, wafers, gt, preds, fab.start_date, METHOD, REVIEW_BUFFERS)


def write_summary(synth: pd.DataFrame, real: dict | None, path: Path) -> None:
    L = ["# E5 요약: 개선 방법의 격리 여유 시간", "",
         "> `scripts/run_e5.py`가 자동 생성한다. 해석은 실험 기록 E5에 있다.", "",
         f"방법 `{METHOD}`, 이력 seed {DEV_SEED}, 합성 예측 {len(synth)}개.", "",
         f"판단 기준: 실제 예측 lot recall 1.0, 합성 lot recall 평균 {MIN_SYNTH_RECALL_MEAN} 이상, "
         f"최솟값 {MIN_SYNTH_RECALL_MIN} 이상인 값 중 가장 짧은 값. {CURRENT_BUFFER}시간보다 짧지 않으면 유지.", ""]
    if real is None:
        L += ["실제 예측을 돌리지 않았다(`--skip-real`). 결론은 실제 예측과 함께 낸다.", "",
              "| 여유 시간 | 합성 recall 평균 | 합성 recall 최소 | 합성 precision 평균 |", "| --- | --- | --- | --- |"]
        for b in REVIEW_BUFFERS:
            r, p = synth[f"lot_recall_{b}h"], synth[f"lot_precision_{b}h"]
            L.append(f"| {b}h | {r.mean():.3f} | {r.min():.3f} | {p.mean():.3f} |")
    else:
        chosen, table = choose_buffer(real, synth)
        L += [f"**선택한 여유 시간: {chosen}시간**", "",
              "| 여유 시간 | 실제 recall | 실제 precision | 합성 recall 평균 | 합성 recall 최소 | 합성 precision 평균 | 기준 통과 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in table.to_dict("records"):
            L.append(f"| {r['buffer_hours']}h | {r['real_lot_recall']:.3f} | {r['real_lot_precision']:.3f} "
                     f"| {r['synth_recall_mean']:.3f} | {r['synth_recall_min']:.3f} | {r['synth_precision_mean']:.3f} "
                     f"| {'O' if r['passes'] else 'X'} |")
    L += ["", f"합성 예측 감지 {int(synth['detected'].sum())}/{len(synth)}, "
          f"원인 특정 {int(synth['cause_found'].sum())}/{len(synth)}."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--skip-real", action="store_true", help="실제 예측 생략 (분류기와 데이터 불필요)")
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/s1.yaml"))
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/cnn_v1.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/reports"))
    args = parser.parse_args()

    fab = load_fab_config(args.fab)
    sc = load_scenario(args.scenario, fab)
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    synth = run_synthetic(fab, sc, args.draws)
    synth.to_csv(args.out / "e5_synthetic.csv", index=False)
    print(f"합성 예측 {args.draws}개 완료 ({time.time() - t:.0f}s)")

    real = None
    if not args.skip_real:
        t = time.time()
        real = run_real(args)
        pd.DataFrame([real]).to_csv(args.out / "e5_real.csv", index=False)
        print(f"실제 예측 seed {DEV_SEED} 완료 ({time.time() - t:.0f}s)")

    write_summary(synth, real, args.out / "e5_summary.md")
    if real is not None:
        chosen, _ = choose_buffer(real, synth)
        print(f"선택한 여유 시간: {chosen}시간")
    print(f"  summary: {args.out / 'e5_summary.md'}")


if __name__ == "__main__":
    main()
