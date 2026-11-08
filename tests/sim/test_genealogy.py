import pandas as pd

from fabtrace.sim.genealogy import generate_genealogy


def test_counts(fab, genealogy):
    n_lots = fab.days * fab.lots_per_day
    assert genealogy["lot_id"].nunique() == n_lots
    assert genealogy["wafer_id"].nunique() == n_lots * fab.wafers_per_lot


def test_one_row_per_wafer_step(fab, genealogy):
    per_wafer = genealogy.groupby("wafer_id")["step"].agg(["size", "nunique"])
    assert (per_wafer["size"] == len(fab.steps)).all()
    assert (per_wafer["nunique"] == len(fab.steps)).all()


def test_chamber_belongs_to_tool(fab, genealogy):
    assert set(genealogy["chamber"]) <= fab.chamber_ids()
    assert (genealogy["chamber"].str.split("/").str[0] == genealogy["tool"]).all()


def test_lot_uses_one_tool_per_step(genealogy):
    assert (genealogy.groupby(["lot_id", "step"])["tool"].nunique() == 1).all()


def test_every_wafer_passes_bottleneck(fab, genealogy):
    for tool in fab.bottleneck_tools():
        passed = genealogy.loc[genealogy["tool"] == tool, "wafer_id"].nunique()
        assert passed == genealogy["wafer_id"].nunique()


def test_timestamps_increase_by_step(genealogy):
    ordered = genealogy.sort_values(["wafer_id", "step_seq"])
    assert (ordered.groupby("wafer_id")["ts"].diff().dropna() > pd.Timedelta(0)).all()


def test_reproducible(fab, genealogy):
    pd.testing.assert_frame_equal(genealogy, generate_genealogy(fab, seed=42))
    assert not genealogy.equals(generate_genealogy(fab, seed=7))