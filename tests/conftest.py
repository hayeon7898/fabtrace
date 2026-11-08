"""공용 fixture와 데이터 의존 테스트 처리.

- needs_data 표시가 붙은 테스트는 data/interim이 없으면 건너뛴다 (CI 환경).
- 이력 생성처럼 무거운 준비는 세션당 한 번만 한다.
"""

from pathlib import Path

import pandas as pd
import pytest

from fabtrace.data.wm811k import META_FILE
from fabtrace.sim.config import load_fab_config
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "interim"
FAB_PATH = ROOT / "configs" / "fab.yaml"
S1_PATH = ROOT / "configs" / "scenarios" / "s1.yaml"

# 실제 WM-811K 라벨 개수 (README 분포표와 같음)
REAL_CLASS_COUNTS = {
    "none": 147431, "Edge-Ring": 9680, "Edge-Loc": 5189, "Center": 4294, "Loc": 3593,
    "Scratch": 1193, "Random": 866, "Donut": 555, "Near-full": 149,
}


def pytest_collection_modifyitems(config, items):
    if (DATA_DIR / META_FILE).exists():
        return
    skip = pytest.mark.skip(reason="data/interim 없음 (CI 환경)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fab():
    return load_fab_config(FAB_PATH)


@pytest.fixture(scope="session")
def s1(fab):
    return load_scenario(S1_PATH, fab)


@pytest.fixture(scope="session")
def genealogy(fab):
    """seed 42 이력. 여러 테스트가 공유하므로 수정하지 말 것."""
    return generate_genealogy(fab, seed=42)


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory):
    """실제와 같은 클래스 개수의 라벨 표만 담은 가짜 data/interim.

    generate는 맵 이미지가 아니라 라벨 목록만 읽으므로, CI에서도 실제와 같은 경로로 실행된다.
    """
    d = tmp_path_factory.mktemp("interim")
    labels = [c for c, n in REAL_CLASS_COUNTS.items() for _ in range(n)]
    pd.DataFrame({"label": labels}).to_parquet(d / META_FILE, index=False)
    return d


@pytest.fixture(scope="session")
def s1_path():
    return S1_PATH


@pytest.fixture(scope="session")
def real_data_dir():
    return DATA_DIR