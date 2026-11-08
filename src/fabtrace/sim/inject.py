"""시나리오대로 결함을 주입하고 가상 팹 웨이퍼마다 WM-811K 맵을 배정한다.

- 영향 웨이퍼: 기간 안에 원인 챔버를 지난 웨이퍼 (원인 단계 통과 시각 기준)
- 평상시 패턴: 가상 팹 몫(fab pool)의 자연 클래스 분포를 따른다
- 영향 웨이퍼는 intensity 확률로 시나리오 패턴으로 바뀐다
- 맵은 fab pool에서 중복 없이 뽑는다. fab pool은 분류기 학습에 쓰이지 않는다

실행은 fabtrace.sim.generate가 담당한다.
"""

import numpy as np
import pandas as pd

from fabtrace.sim.config import FabConfig
from fabtrace.sim.scenario import Scenario


def affected_wafers(genealogy: pd.DataFrame, sc: Scenario, start_date: str) -> np.ndarray:
    lo, hi = sc.window(start_date)
    hit = (genealogy["chamber"] == sc.chamber) & (genealogy["ts"] >= lo) & (genealogy["ts"] < hi)
    return np.sort(genealogy.loc[hit, "wafer_id"].unique())


def assign_maps(
    wafer_ids: np.ndarray, affected: np.ndarray, sc: Scenario, pool: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """pool의 인덱스는 npz 맵 순서와 같아야 한다 (load_meta 결과를 그대로 거른 것)."""
    rng = np.random.default_rng(seed)
    probs = pool["label"].value_counts(normalize=True).sort_index()
    n = len(wafer_ids)

    pattern = rng.choice(probs.index.to_numpy(), size=n, p=probs.to_numpy()).astype(object)
    is_affected = np.isin(wafer_ids, affected)
    injected = is_affected & (rng.random(n) < sc.intensity)
    pattern[injected] = sc.pattern

    map_index = np.empty(n, dtype=np.int64)
    for cls in sorted(set(pattern)):
        rows = np.flatnonzero(pattern == cls)
        candidates = pool.index[pool["label"] == cls].to_numpy()
        if len(rows) > len(candidates):
            raise ValueError(f"{cls} 맵이 부족함: 필요 {len(rows)}, 보유 {len(candidates)}")
        map_index[rows] = rng.choice(candidates, size=len(rows), replace=False)

    return pd.DataFrame({
        "wafer_id": wafer_ids,
        "map_index": map_index,
        "true_pattern": pattern.astype(str),
        "affected": is_affected,
        "injected": injected,
    })


def build_wafers(genealogy, sc, fab: FabConfig, pool, seed) -> pd.DataFrame:
    first = genealogy.sort_values("step_seq").groupby("wafer_id", sort=True).first()
    day = (first["ts"].dt.normalize() - pd.Timestamp(fab.start_date)).dt.days + 1
    wafers = assign_maps(first.index.to_numpy(), affected_wafers(genealogy, sc, fab.start_date), sc, pool, seed)
    wafers.insert(1, "lot_id", first["lot_id"].to_numpy())
    wafers.insert(2, "day", day.to_numpy())
    return wafers


def ground_truth(sc: Scenario, fab: FabConfig, wafers: pd.DataFrame, seed: int) -> dict:
    lo, hi = sc.window(fab.start_date)
    affected = wafers[wafers["affected"]]
    return {
        "scenario": sc.id,
        "seed": seed,
        "step": sc.step,
        "tool": sc.tool,
        "chamber": sc.chamber,
        "pattern": sc.pattern,
        "intensity": sc.intensity,
        "window": {"start_day": sc.start_day, "end_day": sc.end_day,
                   "start_ts": lo.isoformat(), "end_ts": hi.isoformat()},
        "n_affected_wafers": int(len(affected)),
        "n_injected_wafers": int(wafers["injected"].sum()),
        "affected_lots": sorted(affected["lot_id"].unique().tolist()),
    }