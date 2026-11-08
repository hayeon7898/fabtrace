from pathlib import Path

import pytest

from fabtrace.sim.config import load_fab_config
from fabtrace.sim.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]
FAB = load_fab_config(ROOT / "configs" / "fab.yaml")


def test_s1_loads():
    sc = load_scenario(ROOT / "configs" / "scenarios" / "s1.yaml", FAB)
    assert (sc.chamber, sc.tool, sc.pattern) == ("ETCH-03/B", "ETCH-03", "Edge-Ring")


@pytest.mark.parametrize("bad", [
    "chamber: ETCH-09/B",          # 없는 챔버
    "chamber: CMP-01/A",           # ETCH 단계 설비가 아님
])
def test_invalid_scenario_raises(tmp_path, bad):
    text = (ROOT / "configs" / "scenarios" / "s1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("chamber: ETCH-03/B", bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(path, FAB)