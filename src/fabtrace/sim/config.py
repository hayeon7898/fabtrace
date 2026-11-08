"""configs/fab.yaml을 읽어 공정 구조를 담은 객체로 만든다."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Step:
    name: str
    hours: float
    tools: dict[str, tuple[str, ...]]  # 설비 ID -> 챔버 이름들


@dataclass(frozen=True)
class FabConfig:
    start_date: str
    days: int
    lots_per_day: int
    wafers_per_lot: int
    steps: tuple[Step, ...]

    def chamber_ids(self) -> set[str]:
        """'ETCH-03/B' 형태의 전체 챔버 ID 목록."""
        return {f"{t}/{c}" for s in self.steps for t, cs in s.tools.items() for c in cs}

    def bottleneck_tools(self) -> list[str]:
        """설비가 하나뿐이라 모든 웨이퍼가 거치는 설비."""
        return [next(iter(s.tools)) for s in self.steps if len(s.tools) == 1]


def load_fab_config(path: str | Path) -> FabConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    steps = tuple(
        Step(name=s["name"], hours=float(s["hours"]),
             tools={t: tuple(cs) for t, cs in s["tools"].items()})
        for s in raw["steps"]
    )
    config = FabConfig(
        start_date=str(raw["start_date"]),
        days=int(raw["days"]),
        lots_per_day=int(raw["lots_per_day"]),
        wafers_per_lot=int(raw["wafers_per_lot"]),
        steps=steps,
    )
    _validate(config)
    return config


def _validate(config: FabConfig) -> None:
    names = [s.name for s in config.steps]
    if len(names) != len(set(names)):
        raise ValueError(f"공정 단계 이름이 중복됨: {names}")
    tools = [t for s in config.steps for t in s.tools]
    if len(tools) != len(set(tools)):
        raise ValueError("설비 ID가 여러 단계에 중복됨")
    for s in config.steps:
        if not s.tools or any(len(cs) == 0 for cs in s.tools.values()):
            raise ValueError(f"{s.name}: 설비 또는 챔버가 비어 있음")