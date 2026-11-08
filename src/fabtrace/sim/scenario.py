"""결함 주입 시나리오 YAML을 읽고 가상 팹 구조와 맞는지 검사한다."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from fabtrace.data.wm811k import CLASSES
from fabtrace.sim.config import FabConfig


@dataclass(frozen=True)
class Scenario:
    id: str
    step: str
    chamber: str
    start_day: int
    end_day: int
    pattern: str
    intensity: float

    @property
    def tool(self) -> str:
        return self.chamber.split("/")[0]

    def window(self, start_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """[시작, 끝) 시각. 1일차가 start_date 당일이다."""
        base = pd.Timestamp(start_date)
        return base + pd.Timedelta(days=self.start_day - 1), base + pd.Timedelta(days=self.end_day)


def load_scenario(path: str | Path, fab: FabConfig) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sc = Scenario(
        id=str(raw["id"]),
        step=raw["cause"]["step"],
        chamber=raw["cause"]["chamber"],
        start_day=int(raw["window"]["start_day"]),
        end_day=int(raw["window"]["end_day"]),
        pattern=raw["pattern"],
        intensity=float(raw["intensity"]),
    )
    _validate(sc, fab)
    return sc


def _validate(sc: Scenario, fab: FabConfig) -> None:
    if sc.chamber not in fab.chamber_ids():
        raise ValueError(f"{sc.id}: 가상 팹에 없는 챔버 {sc.chamber}")
    step = next((s for s in fab.steps if s.name == sc.step), None)
    if step is None or sc.tool not in step.tools:
        raise ValueError(f"{sc.id}: {sc.tool}은 {sc.step} 단계 설비가 아님")
    if not 1 <= sc.start_day <= sc.end_day <= fab.days:
        raise ValueError(f"{sc.id}: 기간 {sc.start_day}~{sc.end_day}일차가 범위를 벗어남")
    if sc.pattern not in CLASSES:
        raise ValueError(f"{sc.id}: 알 수 없는 패턴 {sc.pattern}")
    if not 0 < sc.intensity <= 1:
        raise ValueError(f"{sc.id}: 강도는 0 초과 1 이하")