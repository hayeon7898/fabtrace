"""시나리오대로 결함을 주입하고 가상 팹 웨이퍼마다 WM-811K 맵을 배정한다.

- 영향 웨이퍼: 원인마다, 기간 안에 원인 챔버를 지난 웨이퍼 (원인 단계 통과 시각 기준).
  설비 단위 원인은 그 설비의 챔버 전부를 본다.
- 평상시 패턴: 가상 팹 몫(fab pool)의 자연 클래스 분포를 따른다.
- 주입: 원인마다 영향 웨이퍼가 그 원인의 강도 확률로 원인 패턴으로 바뀐다. 강도는 원인 챔버를 지난
  시각의 값(Cause.intensity_at)이다.
- 원인 둘에 걸린 웨이퍼: 시나리오에 적힌 순서대로 원인을 적용하고, 먼저 주입된 패턴을 유지한다.
- 난수 순서: 평상시 패턴, 원인마다 주입 난수 한 번씩, 맵 배정. 원인이 하나이고 강도가 일정하면
  원인 목록 이전과 같은 난수를 같은 순서로 쓴다(S1 결과가 그대로다).
- 맵은 fab pool에서 중복 없이 뽑는다. fab pool은 분류기 학습에 쓰이지 않는다.
- wafers의 affected, injected는 원인 중 하나라도 해당하면 참이다. 원인별 영향 웨이퍼는 cause_passes로
  다시 계산할 수 있고, 원인별 개수와 영향 로트는 ground_truth에 남는다.

실행은 fabtrace.sim.generate가 담당한다.
"""

import numpy as np
import pandas as pd

from fabtrace.sim.config import FabConfig
from fabtrace.sim.scenario import Cause, Scenario


def cause_passes(genealogy: pd.DataFrame, cause: Cause, fab: FabConfig) -> pd.DataFrame:
    """원인 기간 안에 원인 챔버를 지난 기록 (wafer_id, chamber, ts). 웨이퍼 하나가 한 단계에서 한 번 지난다."""
    lo, hi = cause.window(fab.start_date)
    hit = genealogy["chamber"].isin(cause.chambers(fab)) & (genealogy["ts"] >= lo) & (genealogy["ts"] < hi)
    return genealogy.loc[hit, ["wafer_id", "chamber", "ts"]].sort_values("wafer_id").reset_index(drop=True)


def affected_wafers(genealogy: pd.DataFrame, sc: Scenario, fab: FabConfig) -> np.ndarray:
    """원인 중 하나라도 영향을 준 웨이퍼."""
    ids = [cause_passes(genealogy, c, fab)["wafer_id"] for c in sc.causes]
    return np.sort(pd.concat(ids).unique()) if ids else np.array([], dtype=object)


def assign_maps(wafer_ids: np.ndarray, genealogy: pd.DataFrame, sc: Scenario, fab: FabConfig,
                pool: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """(wafers, 원인별 통계). pool의 인덱스는 npz 맵 순서와 같아야 한다 (load_meta 결과를 그대로 거른 것)."""
    rng = np.random.default_rng(seed)
    probs = pool["label"].value_counts(normalize=True).sort_index()
    n = len(wafer_ids)

    pattern = rng.choice(probs.index.to_numpy(), size=n, p=probs.to_numpy()).astype(object)
    affected = np.zeros(n, dtype=bool)
    injected = np.zeros(n, dtype=bool)
    stats = []
    for cause in sc.causes:
        passes = cause_passes(genealogy, cause, fab)
        ts = pd.Series(pd.NaT, index=wafer_ids, dtype=passes["ts"].dtype)
        ts.loc[passes["wafer_id"].to_numpy()] = passes["ts"].to_numpy()
        hit = ts.notna().to_numpy()
        if cause.constant:
            p = np.where(hit, cause.intensity_start, 0.0)
        else:
            p = np.where(hit, cause.intensity_at(ts.fillna(pd.Timestamp(0)), fab.start_date).to_numpy(), 0.0)
        new = hit & (rng.random(n) < p) & ~injected  # 먼저 주입된 웨이퍼는 그대로 둔다
        pattern[new] = cause.pattern
        affected |= hit
        injected |= new
        stats.append({"affected": hit.copy(), "injected": new})

    map_index = np.empty(n, dtype=np.int64)
    for cls in sorted(set(pattern)):
        rows = np.flatnonzero(pattern == cls)
        candidates = pool.index[pool["label"] == cls].to_numpy()
        if len(rows) > len(candidates):
            raise ValueError(f"{cls} 맵이 부족함: 필요 {len(rows)}, 보유 {len(candidates)}")
        map_index[rows] = rng.choice(candidates, size=len(rows), replace=False)

    wafers = pd.DataFrame({
        "wafer_id": wafer_ids,
        "map_index": map_index,
        "true_pattern": pattern.astype(str),
        "affected": affected,
        "injected": injected,
    })
    return wafers, stats


def build_wafers(genealogy, sc, fab: FabConfig, pool, seed, return_stats: bool = False):
    first = genealogy.sort_values("step_seq").groupby("wafer_id", sort=True).first()
    day = (first["ts"].dt.normalize() - pd.Timestamp(fab.start_date)).dt.days + 1
    wafers, stats = assign_maps(first.index.to_numpy(), genealogy, sc, fab, pool, seed)
    wafers.insert(1, "lot_id", first["lot_id"].to_numpy())
    wafers.insert(2, "day", day.to_numpy())
    return (wafers, stats) if return_stats else wafers


def ground_truth(sc: Scenario, fab: FabConfig, wafers: pd.DataFrame, seed: int, genealogy: pd.DataFrame) -> dict:
    """주입한 원인 목록과 영향 로트.

    원인이 챔버 하나이고 강도가 일정하면, 예전 형식의 최상위 키(step, tool, chamber, pattern, intensity,
    window)도 함께 쓴다. 그 밖에는 causes만 쓴다. 영향 웨이퍼 수, 주입 웨이퍼 수, 영향 로트는 모든
    원인의 합집합으로 늘 최상위에 둔다.
    """
    lot_of = wafers.set_index("wafer_id")["lot_id"]
    pattern_of = wafers.set_index("wafer_id")["true_pattern"]
    causes = []
    for c in sc.causes:
        lo, hi = c.window(fab.start_date)
        ids = cause_passes(genealogy, c, fab)["wafer_id"]
        causes.append({
            "unit": c.unit, "step": c.step, "location": c.location, "tool": c.tool,
            "chambers": list(c.chambers(fab)), "pattern": c.pattern,
            "intensity": {"start": c.intensity_start, "end": c.intensity_end},
            "window": {"start_day": c.start_day, "end_day": c.end_day,
                       "start_ts": lo.isoformat(), "end_ts": hi.isoformat()},
            "n_affected_wafers": int(len(ids)),
            "n_pattern_wafers": int((pattern_of.loc[ids] == c.pattern).sum()),
            "affected_lots": sorted(lot_of.loc[ids].unique().tolist()),
        })

    affected = wafers[wafers["affected"]]
    gt = {"scenario": sc.id, "seed": seed}
    if len(sc.causes) == 1 and sc.causes[0].unit == "chamber" and sc.causes[0].constant:
        c, d = sc.causes[0], causes[0]
        gt |= {"step": c.step, "tool": c.tool, "chamber": c.location, "pattern": c.pattern,
               "intensity": c.intensity_start, "window": d["window"]}
    gt |= {
        "n_affected_wafers": int(len(affected)),
        "n_injected_wafers": int(wafers["injected"].sum()),
        "affected_lots": sorted(affected["lot_id"].unique().tolist()),
        "causes": causes,
    }
    return gt
