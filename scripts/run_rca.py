"""가상 팹 생성부터 예측, 관리도, 공통성 분석, 격리 범위까지 한 번에 실행하고 리포트를 만든다.

사용법: uv run python scripts/run_rca.py --scenario s1 --seed 42

출력
  results/reports/s1_rca.md     리포트
  results/reports/s1_rca.json   분석 결과(analysis)와 채점(evaluation). 스펙 05 입력
  results/figures/s1_rca_*.png  p 관리도, 공통성 비교
"""

import argparse
from pathlib import Path

from fabtrace.analysis.containment import DEFAULT_BUFFER_HOURS
from fabtrace.analysis.pipeline import run_analysis, score_result, simulate_and_predict
from fabtrace.analysis.report import write_report
from fabtrace.data.dataset import load_wafer_data


def scenario_path(arg: str) -> Path:
    p = Path(arg)
    return p if p.suffix in (".yaml", ".yml") else Path("configs/scenarios") / f"{arg.lower()}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="s1", help="시나리오 이름(s1) 또는 YAML 경로")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=Path("models/cnn_v1.pt"))
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--buffer-hours", type=float, default=DEFAULT_BUFFER_HOURS)
    parser.add_argument("--data", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    sc_path = scenario_path(args.scenario)
    fab, genealogy, wafers, gt, predictions = simulate_and_predict(
        sc_path, args.fab, args.seed, load_wafer_data(args.data), args.model)
    result = run_analysis(predictions, genealogy, fab.start_date, buffer_hours=args.buffer_hours)
    evaluation = score_result(result, gt, wafers)

    meta = {"scenario": gt["scenario"], "seed": args.seed, "model": args.model.stem,
            "truth_chamber": gt["chamber"], "truth_pattern": gt["pattern"],
            "truth_window": [gt["window"]["start_day"], gt["window"]["end_day"]]}
    name = f"{gt['scenario'].lower()}_rca"
    paths = write_report(result, evaluation, meta, name, Path("results/reports"), Path("results/figures"))

    print(f"{gt['scenario']} seed {args.seed}: 감지 {evaluation['detected']}, 원인 후보 {evaluation.get('causes')}, "
          f"방법 A 순위 {evaluation.get('rank_a')}, 방법 B 순위 {evaluation.get('rank_b')}, "
          f"로트 recall {evaluation.get('lot_recall')}")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()