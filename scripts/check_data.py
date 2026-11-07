"""변환된 WM-811K 파일이 제자리에 있고 온전한지 빠르게 확인한다.

원본 LSWMD.pkl은 메모리를 많이 써서 로컬에서 읽지 않는다.
변환은 scripts/prepare_wm811k.py를 Kaggle 등에서 한 번 실행해 만든다.

사용법: uv run python scripts/check_data.py [--dir data/interim]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_LABELED = 172_950
EXPECTED_CLASSES = {
    "none", "Edge-Ring", "Edge-Loc", "Center", "Loc",
    "Scratch", "Random", "Donut", "Near-full",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("data/interim"))
    d = parser.parse_args().dir

    npz, meta_path = d / "wm811k_labeled_64.npz", d / "wm811k_labeled_64.parquet"
    missing = [p for p in (npz, meta_path) if not p.exists()]
    if missing:
        print("파일 없음: " + ", ".join(map(str, missing)))
        print("README의 데이터 준비 절을 따라 변환 파일을 data/interim/에 넣어주세요.")
        return 1

    maps = np.load(npz)["maps"]
    meta = pd.read_parquet(meta_path)

    checks = {
        f"라벨 맵 수 {EXPECTED_LABELED:,}개": len(meta) == EXPECTED_LABELED,
        "맵과 메타데이터 행 수 일치": maps.shape[0] == len(meta),
        "맵 크기 64×64": maps.shape[1:] == (64, 64),
        "다이 값이 0/1/2뿐": set(np.unique(maps)) <= {0, 1, 2},
        "클래스 9종": set(meta["label"]) == EXPECTED_CLASSES,
    }
    for name, ok in checks.items():
        print(f"[{'OK' if ok else '실패'}] {name}")
    print(meta["label"].value_counts().to_string())
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())