"""WM-811K 원본(LSWMD.pkl)을 한 번만 읽어 가벼운 학습용 파일로 변환한다.

메모리가 넉넉한 곳(Kaggle 노트북 등)에서 한 번 실행하고, 결과 파일만 로컬 data/interim/에 둔다.

사용법
  Kaggle 노트북: 데이터셋을 Input으로 추가한 뒤 이 파일 내용을 셀에 붙여 실행
  로컬:          uv run python scripts/prepare_wm811k.py --input data/raw/LSWMD.pkl --out data/interim

출력
  wm811k_labeled_64.npz   maps: (N, 64, 64) uint8, 값 0=웨이퍼 밖, 1=정상 다이, 2=불량 다이
  wm811k_labeled_64.parquet  행 순서가 maps와 같은 메타데이터
  wm811k_samples.png      클래스별 예시 맵 (README용)
"""

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIZE = 64
EXPECTED_ROWS = 811_457


def find_input() -> Path:
    hits = glob.glob("/kaggle/input/**/LSWMD.pkl", recursive=True)
    if not hits:
        sys.exit("LSWMD.pkl을 찾지 못했습니다. --input으로 경로를 지정하세요.")
    return Path(hits[0])


def load_lswmd(path: Path) -> pd.DataFrame:
    """파이썬 2와 옛 pandas로 저장된 파일이라 latin1 인코딩과 옛 모듈 경로 처리가 필요하다."""
    import pandas.core.indexes as indexes

    sys.modules.setdefault("pandas.indexes", indexes)
    try:
        return pd.read_pickle(path)
    except UnicodeDecodeError:
        from pandas.compat.pickle_compat import Unpickler

        with path.open("rb") as f:
            return Unpickler(f, encoding="latin1").load()


def first_str(value):
    """[['Edge-Ring']] 같은 중첩 배열에서 문자열 하나를 꺼낸다. 비어 있으면 None."""
    arr = np.asarray(value)
    return str(arr.ravel()[0]) if arr.size else None


def resize_nearest(m: np.ndarray, size: int = SIZE) -> np.ndarray:
    """최근접 보간. 다이 값 0/1/2가 섞이지 않도록 보간 없이 인덱스만 고른다."""
    h, w = m.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return m[np.ix_(rows, cols)].astype(np.uint8)


def save_samples(maps: np.ndarray, labels: pd.Series, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(labels.unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(2 * len(classes), 2.4))
    for ax, cls in zip(np.atleast_1d(axes), classes):
        ax.imshow(maps[labels.index[labels == cls][0]], cmap="viridis", vmin=0, vmax=2)
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "wm811k_samples.png", dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    args, _ = parser.parse_known_args()  # 노트북 커널 인자는 무시

    src = args.input or find_input()
    out = args.out or (Path("/kaggle/working") if Path("/kaggle").exists() else Path("data/interim"))
    out.mkdir(parents=True, exist_ok=True)

    df = load_lswmd(src)
    print(f"전체 웨이퍼 맵: {len(df):,}개 (기대값 {EXPECTED_ROWS:,})")

    df["label"] = df["failureType"].map(first_str)
    df["split"] = df["trianTestLabel"].map(first_str)  # 원본 컬럼명 오타 그대로
    labeled = df[df["label"].notna()].reset_index().rename(columns={"index": "orig_index"})
    print(f"라벨 있음: {len(labeled):,}개")

    maps = np.stack([resize_nearest(np.asarray(m)) for m in labeled["waferMap"]])
    shapes = labeled["waferMap"].map(lambda m: np.asarray(m).shape)

    meta = pd.DataFrame(
        {
            "orig_index": labeled["orig_index"],
            "label": labeled["label"],
            "split": labeled["split"],
            "lot_name": labeled["lotName"].astype(str),
            "wafer_index": labeled["waferIndex"].astype(float).astype(int),
            "orig_h": shapes.map(lambda s: s[0]),
            "orig_w": shapes.map(lambda s: s[1]),
        }
    )

    np.savez_compressed(out / "wm811k_labeled_64.npz", maps=maps)
    meta.to_parquet(out / "wm811k_labeled_64.parquet", index=False)
    save_samples(maps, meta["label"], out)

    assert maps.shape == (len(meta), SIZE, SIZE)
    assert set(np.unique(maps)) <= {0, 1, 2}
    print(meta["label"].value_counts().to_string())
    for f in sorted(out.glob("wm811k_*")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()