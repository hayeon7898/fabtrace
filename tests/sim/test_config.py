import pytest

from fabtrace.sim.config import load_fab_config


def test_structure(fab):
    assert len(fab.steps) == 6
    assert "ETCH-03/B" in fab.chamber_ids()
    assert fab.bottleneck_tools() == ["CLN-01"]


def test_duplicate_tool_raises(tmp_path):
    path = tmp_path / "fab.yaml"
    path.write_text(
        "start_date: '2026-01-01'\ndays: 1\nlots_per_day: 1\nwafers_per_lot: 1\n"
        "steps:\n"
        "  - {name: A, hours: 1, tools: {T-01: [A]}}\n"
        "  - {name: B, hours: 1, tools: {T-01: [A]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_fab_config(path)