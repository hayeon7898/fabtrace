"""웨이퍼별 공정 이력(어느 단계에서 어느 설비, 챔버를 언제 거쳤는지)을 만든다.

배정 규칙
- 로트는 단계마다 설비 하나를 무작위로 골라 통째로 들어간다.
- 로트 안 웨이퍼들은 그 설비의 챔버들에 슬롯 순서대로 나뉜다.

시간 규칙
- 로트는 하루 동안 고르게 투입되고 약간의 흔들림을 준다.
- 각 단계는 설정된 소요 시간에 0~2시간 대기를 더한다.
- 같은 로트 안에서는 슬롯마다 몇 분씩 차이를 둔다.

실행은 fabtrace.sim.generate가 담당한다.
"""

import numpy as np
import pandas as pd

from fabtrace.sim.config import FabConfig

SLOT_MINUTES = 3
MAX_WAIT_HOURS = 2.0
LOT_JITTER_MINUTES = 30


def generate_genealogy(config: FabConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.start_date)
    lot_gap = pd.Timedelta(hours=24 / config.lots_per_day)
    slots = np.arange(1, config.wafers_per_lot + 1)
    slot_offset = pd.to_timedelta((slots - 1) * SLOT_MINUTES, unit="m")

    frames = []
    n_lots = config.days * config.lots_per_day
    for i in range(n_lots):
        lot_id = f"L{i + 1:04d}"
        jitter = pd.Timedelta(minutes=float(rng.uniform(-LOT_JITTER_MINUTES, LOT_JITTER_MINUTES)))
        step_start = start + i * lot_gap + jitter

        for seq, step in enumerate(config.steps, start=1):
            tool_ids = list(step.tools)
            tool = tool_ids[rng.integers(len(tool_ids))]
            chambers = step.tools[tool]
            first = rng.integers(len(chambers))
            assigned = [f"{tool}/{chambers[(first + s) % len(chambers)]}" for s in range(len(slots))]

            frames.append(pd.DataFrame({
                "wafer_id": [f"{lot_id}-W{s:02d}" for s in slots],
                "lot_id": lot_id,
                "slot": slots,
                "step_seq": seq,
                "step": step.name,
                "tool": tool,
                "chamber": assigned,
                "ts": (step_start + slot_offset).round("s"),
            }))
            wait = pd.Timedelta(hours=float(rng.uniform(0, MAX_WAIT_HOURS)))
            step_start += pd.Timedelta(hours=step.hours) + wait

    return pd.concat(frames, ignore_index=True)