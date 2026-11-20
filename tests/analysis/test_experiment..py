import pandas as pd
import pytest

from fabtrace.analysis.experiment import BUFFERS, evaluate_seed, summarize
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth


def _rows(fab, s1, method):
    counts = {"none": 11000, "Edge-Ring": 800, "Edge-Loc": 400, "Center": 350, "Loc": 300}
    labels = [c for c, n in counts.items() for _ in range(n)]
    pool = pd.DataFrame({"label": labels}, index=range(len(labels)))
    out = []
    for seed in (0, 1):
        gen = generate_genealogy(fab, seed)
        wafers = build_wafers(gen, s1, fab, pool, seed)
        preds = wafers[["wafer_id"]].assign(pred_pattern=wafers["true_pattern"])
        out.append(evaluate_seed(seed, gen, wafers, ground_truth(s1, fab, wafers, seed), preds, fab.start_date,
                                 method=method))
    return pd.DataFrame(out)


@pytest.fixture(scope="module")
def rows(fab, s1):
    return _rows(fab, s1, "baseline")


@pytest.fixture(scope="module")
def rows_exposure(fab, s1):
    return _rows(fab, s1, "exposure")


def test_rows_have_all_metrics(rows):
    for b in BUFFERS:
        assert {f"lot_recall_{b}h", f"lot_precision_{b}h", f"wafer_recall_{b}h"} <= set(rows.columns)
    assert rows["detected"].all() and rows["cause_found"].all()
    assert (rows["rank_b"] == 1).all()


def test_more_buffer_never_lowers_recall(rows):
    assert (rows["lot_recall_24h"] >= rows["lot_recall_0h"]).all()


def test_summary(rows):
    s = summarize(rows)
    assert s["seeds"] == 2 and s["top1_b"] == "2/2"

def test_exposure_method_reaches_buffer_columns(rows, rows_exposure):
    """여유 시간별 격리 수치도 method를 따른다. 감지와 원인 순위는 같다."""
    assert rows_exposure[["detected", "rank_a", "rank_b"]].equals(rows[["detected", "rank_a", "rank_b"]])
    for b in BUFFERS:
        assert (rows_exposure[f"lot_recall_{b}h"] >= rows[f"lot_recall_{b}h"]).all()
    assert (rows_exposure["lot_recall_0h"] == 1.0).all()
