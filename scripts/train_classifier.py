"""웨이퍼 맵 분류기 학습.

사용법
  uv run python scripts/train_classifier.py --seed 42                 # 불균형 대응 (기본)
  uv run python scripts/train_classifier.py --seed 42 --no-balance    # 비교용
  uv run python scripts/train_classifier.py --max-epochs 1            # 에폭 시간 측정

출력
  models/cnn_v1.pt (또는 cnn_v1_nobalance.pt)   가중치와 학습 설정, 기록
  results/logs/cnn_v1.csv                  에폭별 train loss, val macro-F1, 시간
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from fabtrace.data.dataset import load_wafer_data
from fabtrace.models.train import TrainConfig, fit

try:  # 같은 scripts/ 폴더의 평가 스크립트
    from evaluate_classifier import evaluate_all
except ImportError:  # pragma: no cover
    from scripts.evaluate_classifier import evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-balance", action="store_true", help="none 샘플링과 클래스 가중치를 끈 비교용 학습")
    parser.add_argument("--threads", type=int, help="torch CPU 스레드 수 (기본: torch 기본값)")
    parser.add_argument("--data", type=Path, default=Path("data/interim"))
    parser.add_argument("--name", help="출력 이름 (기본: cnn_v1 또는 cnn_v1_nobalance)")
    parser.add_argument("--skip-eval", action="store_true", help="학습 후 평가와 예측 저장을 건너뜀")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    name = args.name or ("cnn_v1_nobalance" if args.no_balance else "cnn_v1")
    cfg = TrainConfig(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
                      batch_size=args.batch_size, balance=not args.no_balance)
    print(f"{name}: {cfg}  (torch 스레드 {torch.get_num_threads()})")

    ckpt = fit(load_wafer_data(args.data), cfg)

    Path("models").mkdir(exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, Path("models") / f"{name}.pt")
    pd.DataFrame(ckpt["history"]).to_csv(Path("results/logs") / f"{name}.csv", index=False)
    print(f"최고 val macro-F1 {ckpt['best_val_macro_f1']:.4f} (epoch {ckpt['best_epoch']}) -> models/{name}.pt")

    if not args.skip_eval:
        evaluate_all()


if __name__ == "__main__":
    main()