from pathlib import Path

import pandas as pd
import pytest

from fabtrace.sim.config import load_fab_config
from fabtrace.sim.genealogy import generate_genealogy

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fab.yaml"


@pytest.fixture(scope="module")
def config():
    return load_fab_config(CONFIG)


@pytest.fixture(scope="module")
def gen(config):
    return generate_genealogy(config, seed=42)


def test_config_structure(config):
    assert len(config.steps) == 6
    assert "ETCH-03/B" in config.chamber_ids()
    assert config.bottleneck_tools() == ["CLN-01"]


def test_counts(config, gen):
    n_lots = config.days * config.lots_per_day
    assert gen["lot_id"].nunique() == n_lots
    assert gen["wafer_id"].nunique() == n_lots * config.wafers_per_lot


def test_one_row_per_wafer_step(config, gen):
    per_wafer = gen.groupby("wafer_id")["step"].agg(["size", "nunique"])
    assert (per_wafer["size"] == len(config.steps)).all()
    assert (per_wafer["nunique"] == len(config.steps)).all()


def test_chamber_belongs_to_tool(config, gen):
    assert set(gen["chamber"]) <= config.chamber_ids()
    assert (gen["chamber"].str.split("/").str[0] == gen["tool"]).all()


def test_lot_uses_one_tool_per_step(gen):
    assert (gen.groupby(["lot_id", "step"])["tool"].nunique() == 1).all()


def test_every_wafer_passes_bottleneck(config, gen):
    for tool in config.bottleneck_tools():
        assert gen.loc[gen["tool"] == tool, "wafer_id"].nunique() == gen["wafer_id"].nunique()


def test_timestamps_increase_by_step(gen):
    ordered = gen.sort_values(["wafer_id", "step_seq"])
    assert (ordered.groupby("wafer_id")["ts"].diff().dropna() > pd.Timedelta(0)).all()


def test_reproducible(config, gen):
    pd.testing.assert_frame_equal(gen, generate_genealogy(config, seed=42))
    assert not gen.equals(generate_genealogy(config, seed=7))