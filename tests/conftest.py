"""데이터가 필요한 테스트는 @pytest.mark.needs_data를 붙인다.

CI에는 data/가 없으므로 해당 테스트는 자동으로 건너뛴다.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: data/interim 파일이 있어야 실행되는 테스트")


def pytest_collection_modifyitems(config, items):
    if (DATA_DIR / "wm811k_labeled_64.parquet").exists():
        return
    skip = pytest.mark.skip(reason="data/interim 없음 (CI 환경)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)