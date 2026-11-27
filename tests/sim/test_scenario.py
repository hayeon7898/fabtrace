import pandas as pd
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

def _write(tmp_path, text: str):
    path = tmp_path / "sc.yaml"
    path.write_text(text, encoding="utf-8")
    return path


CHAMBER_CAUSE = """
  - unit: chamber
    step: ETCH
    location: ETCH-03/B
    window: {start_day: 12, end_day: 18}
    pattern: Edge-Ring
    intensity: 0.4"""
TOOL_CAUSE = """
  - unit: tool
    step: ETCH
    location: ETCH-02
    window: {start_day: 12, end_day: 18}
    pattern: Edge-Ring
    intensity: 0.2"""
LINEAR_CAUSE = """
  - unit: chamber
    step: DEPOSITION
    location: DEP-02/A
    window: {start_day: 10, end_day: 24}
    pattern: Edge-Loc
    intensity: {start: 0.0, end: 0.4}"""


def test_s1_is_one_chamber_cause(s1):
    (c,) = s1.causes
    assert (c.unit, c.step, c.location, c.start_day, c.end_day) == ("chamber", "ETCH", "ETCH-03/B", 12, 18)
    assert c.constant and s1.intensity == 0.4


def test_new_format_matches_legacy_s1(tmp_path, fab, s1):
    sc = load_scenario(_write(tmp_path, "id: S1\ncauses:" + CHAMBER_CAUSE), fab)
    assert sc == s1


def test_tool_cause_expands_to_its_chambers(tmp_path, fab):
    sc = load_scenario(_write(tmp_path, "id: S2\ncauses:" + TOOL_CAUSE), fab)
    (c,) = sc.causes
    step = next(s for s in fab.steps if s.name == "ETCH")
    assert c.chambers(fab) == tuple(f"ETCH-02/{ch}" for ch in step.tools["ETCH-02"])
    assert sc.tool == "ETCH-02"
    with pytest.raises(ValueError):
        _ = sc.chamber  # 설비 단위는 챔버 하나로 나타낼 수 없다


def test_two_causes_block_single_cause_attributes(tmp_path, fab):
    sc = load_scenario(_write(tmp_path, "id: S3\ncauses:" + CHAMBER_CAUSE + LINEAR_CAUSE), fab)
    assert [c.location for c in sc.causes] == ["ETCH-03/B", "DEP-02/A"]
    for attr in ("chamber", "pattern", "step"):
        with pytest.raises(ValueError):
            getattr(sc, attr)


def test_no_cause_needs_explicit_empty_list(tmp_path, fab):
    assert load_scenario(_write(tmp_path, "id: S5\ncauses: []"), fab).causes == ()
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, "id: S5"), fab)


def test_linear_intensity(tmp_path, fab):
    sc = load_scenario(_write(tmp_path, "id: S4\ncauses:" + LINEAR_CAUSE), fab)
    (c,) = sc.causes
    lo, hi = c.window(fab.start_date)
    mid = lo + (hi - lo) / 2
    assert not c.constant
    assert c.intensity_at(lo, fab.start_date) == 0.0
    assert c.intensity_at(mid, fab.start_date) == pytest.approx(0.2)
    assert c.intensity_at(hi - pd.Timedelta(seconds=1), fab.start_date) == pytest.approx(0.4, abs=1e-5)
    assert c.intensity_at(hi, fab.start_date) == 0.0  # 기간 밖
    s = c.intensity_at(pd.Series([lo - pd.Timedelta(hours=1), mid]), fab.start_date)
    assert s.tolist() == pytest.approx([0.0, 0.2])
    with pytest.raises(ValueError):
        _ = sc.intensity  # 일정하지 않은 강도는 숫자 하나로 나타낼 수 없다


@pytest.mark.parametrize("bad", [
    CHAMBER_CAUSE.replace("unit: chamber", "unit: lot"),                   # 없는 단위
    CHAMBER_CAUSE.replace("ETCH-03/B", "ETCH-03"),                        # 챔버 단위인데 설비
    TOOL_CAUSE.replace("location: ETCH-02", "location: ETCH-02/A"),       # 설비 단위인데 챔버
    TOOL_CAUSE.replace("ETCH-02", "CMP-01"),                              # ETCH 단계 설비가 아님
    CHAMBER_CAUSE.replace("end_day: 18", "end_day: 99"),                  # 기간 범위
    CHAMBER_CAUSE.replace("Edge-Ring", "none"),                           # 불량 패턴이 아님
    CHAMBER_CAUSE.replace("intensity: 0.4", "intensity: 1.5"),            # 강도 범위
    CHAMBER_CAUSE.replace("intensity: 0.4", "intensity: 0"),              # 강도 0
    LINEAR_CAUSE.replace("{start: 0.0, end: 0.4}", "{start: 0.0, stop: 0.4}"),  # 선형 키
])
def test_invalid_cause_raises(tmp_path, fab, bad):
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, "id: X\ncauses:" + bad), fab)


def test_cause_and_causes_together_raise(tmp_path, fab, s1_path):
    text = s1_path.read_text(encoding="utf-8") + "causes:" + CHAMBER_CAUSE
    with pytest.raises(ValueError):
        load_scenario(_write(tmp_path, text), fab)
