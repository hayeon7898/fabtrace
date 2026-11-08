import pandas as pd
import pytest

from fabtrace.data.wm811k import assign_pools, load_meta


@pytest.fixture
def labels():
    counts = {"none": 6000, "Edge-Ring": 400, "Center": 250, "Near-full": 7}
    return pd.Series([c for c, n in counts.items() for _ in range(n)])


def test_pools_are_stratified(labels):
    pools = assign_pools(labels)
    share = pd.crosstab(labels, pools, normalize="index")
    assert ((share["fab"] - 0.3).abs() < 0.02)[["none", "Edge-Ring", "Center"]].all()
    assert set(pools) == {"train", "val", "fab"}


def test_pools_are_deterministic(labels):
    assert assign_pools(labels).equals(assign_pools(labels))


@pytest.mark.needs_data
def test_real_fab_pool_size():
    meta = load_meta()
    pool = meta[assign_pools(meta["label"]) == "fab"]
    assert 0.29 < len(pool) / len(meta) < 0.31
    assert (pool["label"] == "Edge-Ring").sum() > 2500