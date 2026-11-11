"""저장된 분류기를 fab 몫으로 평가하고, 리포트와 가상 팹 예측을 만든다. 재학습 없이 몇 초면 끝난다.

사용법: uv run python scripts/evaluate_classifier.py

출력
  results/classifier_report.md                 자동 생성 리포트
  results/figures/confusion_<모델>.png          혼동행렬
  results/figures/classifier_learning_curves.png
  data/processed/predictions.parquet           가상 팹 예측 (--predict-with 모델)
"""

import argparse
from pathlib import Path

import pandas as pd

from fabtrace.data.dataset import load_wafer_data
from fabtrace.models.evaluate import run_evaluation

DEFAULT_MODELS = [Path("models/cnn_v1.pt"), Path("models/cnn_v1_nobalance.pt")]
DEFAULT_PREDICT_WITH = Path("models/cnn_v1.pt")  # val macro-F1로 미리 고른 모델


def evaluate_all(models: list[Path] | None = None, predict_with: Path | None = DEFAULT_PREDICT_WITH,
                 data_dir: Path = Path("data/interim"), processed_dir: Path = Path("data/processed"),
                 results_dir: Path = Path("results")) -> None:
    models = [m for m in (models or DEFAULT_MODELS) if m.exists()]
    if not models:
        raise SystemExit("평가할 모델이 없습니다. 먼저 scripts/train_classifier.py를 실행하세요.")
    wafers_path = processed_dir / "wafers.parquet"
    wafers = pd.read_parquet(wafers_path) if wafers_path.exists() else None
    if predict_with is not None and not predict_with.exists():
        predict_with = None

    evals = run_evaluation(models, load_wafer_data(data_dir), wafers, predict_with, results_dir, processed_dir)
    for ev in evals:
        print(f"{ev.name}: fab macro-F1 {ev.macro_f1:.4f}, Edge-Ring recall {ev.per_class.loc['Edge-Ring', 'recall']:.4f}")
    if wafers is None:
        print("wafers.parquet가 없어 가상 팹 예측은 건너뜀 (fabtrace.sim.generate를 먼저 실행)")
    elif predict_with is not None:
        print(f"가상 팹 예측 {len(wafers):,}장 -> {processed_dir / 'predictions.parquet'} ({predict_with.stem})")
    print(f"리포트 -> {results_dir / 'classifier_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+")
    parser.add_argument("--predict-with", type=Path, default=DEFAULT_PREDICT_WITH)
    args = parser.parse_args()
    evaluate_all(args.models, args.predict_with)


if __name__ == "__main__":
    main()