import pytest

from fabtrace.sim.scenario import load_scenario

S1_TEXT_CHAMBER = "chamber: ETCH-03/B"


def test_s1_loads(s1):
    assert (s1.chamber, s1.tool, s1.pattern) == ("ETCH-03/B", "ETCH-03", "Edge-Ring")


@pytest.mark.parametrize("bad", [
    "chamber: ETCH-09/B",  # 없는 챔버
    "chamber: CMP-01/A",   # ETCH 단계 설비가 아님
])
def test_invalid_scenario_raises(tmp_path, fab, s1_path, bad):
    path = tmp_path / "bad.yaml"
    path.write_text(s1_path.read_text(encoding="utf-8").replace(S1_TEXT_CHAMBER, bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(path, fab)