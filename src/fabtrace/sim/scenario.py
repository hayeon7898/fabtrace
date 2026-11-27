"""결함 주입 시나리오 YAML을 읽고 가상 팹 구조와 맞는지 검사한다.

시나리오는 원인 목록이다. 원인마다 단위(챔버 또는 설비), 위치, 기간, 패턴, 강도를 둔다.

새 형식:

    id: S2
    causes:
      - unit: tool               # chamber 또는 tool
        step: ETCH
        location: ETCH-02        # chamber면 ETCH-03/B 형식
        window: {start_day: 12, end_day: 18}   # 원인 단계를 지난 시각 기준, 양 끝 날짜 포함
        pattern: Edge-Ring
        intensity: 0.4           # 숫자 하나면 일정
      - unit: chamber
        step: DEP
        location: DEP-02/A
        window: {start_day: 10, end_day: 24}
        pattern: Edge-Loc
        intensity: {start: 0.0, end: 0.4}      # 기간 시작부터 끝까지 선형

원인이 없는 시나리오는 `causes: []`를 명시한다. 키가 빠지면 에러다.

기존 형식(`cause:`, `window:`, `pattern:`, `intensity:`를 최상위에 둔 원인 하나, 예: s1.yaml)도 그대로
읽는다. 원인이 하나인 시나리오는 Scenario.chamber, .step, .window 등 예전 속성으로도 읽을 수 있다.
원인이 여럿이거나 없으면 이 속성들은 에러를 낸다. 첫 원인만 쓰고 넘어가는 실수를 막기 위해서다.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import yaml

from fabtrace.data.wm811k import CLASSES
from fabtrace.sim.config import FabConfig

UNITS = ("chamber", "tool")
# 난이도 곡선에 쓰는 강도 목록. 에이전트 결과를 보기 전에 정한 값이다(스펙 05 E6, 스펙 06 E7).
INTENSITY_SWEEP = (0.1, 0.15, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class Cause:
    unit: str                 # chamber 또는 tool
    step: str
    location: str             # chamber면 "TOOL/CH", tool이면 "TOOL"
    start_day: int
    end_day: int
    pattern: str
    intensity_start: float
    intensity_end: float      # 일정한 강도면 intensity_start와 같다

    @property
    def tool(self) -> str:
        return self.location.split("/")[0]

    @property
    def constant(self) -> bool:
        return self.intensity_start == self.intensity_end

    def window(self, start_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """[시작, 끝) 시각. 1일차가 start_date 당일이다."""
        base = pd.Timestamp(start_date)
        return base + pd.Timedelta(days=self.start_day - 1), base + pd.Timedelta(days=self.end_day)

    def chambers(self, fab: FabConfig) -> tuple[str, ...]:
        """원인 챔버 ID들. 설비 단위면 그 설비의 챔버 전부다."""
        if self.unit == "chamber":
            return (self.location,)
        step = next(s for s in fab.steps if s.name == self.step)
        return tuple(f"{self.location}/{c}" for c in step.tools[self.location])

    def intensity_at(self, ts, start_date: str):
        """시각 ts(하나 또는 Series)의 강도. 기간 밖이면 0이다. 선형이면 기간 시작에서 끝까지 고르게 변한다."""
        lo, hi = self.window(start_date)
        t = pd.to_datetime(ts)
        frac = (t - lo) / (hi - lo)
        inside = (t >= lo) & (t < hi)
        value = self.intensity_start + (self.intensity_end - self.intensity_start) * frac
        if isinstance(t, pd.Series):
            return value.where(inside, 0.0).astype(float)
        return float(value) if inside else 0.0


@dataclass(frozen=True)
class Scenario:
    id: str
    causes: tuple[Cause, ...]

    # 원인이 하나인 시나리오를 위한 예전 속성. 원인이 여럿이거나 없으면 에러를 낸다.
    @property
    def only_cause(self) -> Cause:
        if len(self.causes) != 1:
            raise ValueError(f"{self.id}: 원인이 {len(self.causes)}개다. causes로 원인마다 다뤄야 한다")
        return self.causes[0]

    @property
    def step(self) -> str:
        return self.only_cause.step

    @property
    def chamber(self) -> str:
        c = self.only_cause
        if c.unit != "chamber":
            raise ValueError(f"{self.id}: 원인이 설비 단위({c.location})라 챔버 하나로 나타낼 수 없다")
        return c.location

    @property
    def tool(self) -> str:
        return self.only_cause.tool

    @property
    def start_day(self) -> int:
        return self.only_cause.start_day

    @property
    def end_day(self) -> int:
        return self.only_cause.end_day

    @property
    def pattern(self) -> str:
        return self.only_cause.pattern

    @property
    def intensity(self) -> float:
        c = self.only_cause
        if not c.constant:
            raise ValueError(f"{self.id}: 강도가 일정하지 않다. intensity_at을 써야 한다")
        return c.intensity_start

    def window(self, start_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.only_cause.window(start_date)


def with_intensity(sc: Scenario, intensity: float) -> Scenario:
    """모든 원인의 강도를 intensity로 바꾼 시나리오. 강도가 일정한 원인만 받는다. id에 강도를 붙인다."""
    for c in sc.causes:
        if not c.constant:
            raise ValueError(f"{sc.id}: 강도가 일정하지 않은 원인({c.location})은 바꿀 수 없다")
    if not 0 < intensity <= 1:
        raise ValueError(f"{sc.id}: 강도는 0 초과 1 이하")
    causes = tuple(replace(c, intensity_start=intensity, intensity_end=intensity) for c in sc.causes)
    return Scenario(id=f"{sc.id}@{intensity:g}", causes=causes)


def load_scenario(path: str | Path, fab: FabConfig) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sid = str(raw["id"])
    if "causes" in raw:
        if "cause" in raw:
            raise ValueError(f"{sid}: cause와 causes를 함께 쓸 수 없다")
        if not isinstance(raw["causes"], list):
            raise ValueError(f"{sid}: causes는 목록이어야 한다 (원인이 없으면 causes: [])")
        causes = tuple(_parse_cause(sid, c) for c in raw["causes"])
    elif "cause" in raw:
        causes = (_parse_legacy(raw),)
    else:
        raise ValueError(f"{sid}: causes가 없다. 원인이 없는 시나리오는 causes: []로 명시한다")
    sc = Scenario(id=sid, causes=causes)
    for c in sc.causes:
        _validate(sid, c, fab)
    return sc


def _parse_legacy(raw: dict) -> Cause:
    chamber = raw["cause"]["chamber"]
    intensity = float(raw["intensity"])
    return Cause(unit="chamber", step=raw["cause"]["step"], location=chamber,
                 start_day=int(raw["window"]["start_day"]), end_day=int(raw["window"]["end_day"]),
                 pattern=raw["pattern"], intensity_start=intensity, intensity_end=intensity)


def _parse_cause(sid: str, c: dict) -> Cause:
    unit = c.get("unit")
    if unit not in UNITS:
        raise ValueError(f"{sid}: 원인 단위는 {UNITS} 중 하나 ({unit!r})")
    intensity = c["intensity"]
    if isinstance(intensity, dict):
        if set(intensity) != {"start", "end"}:
            raise ValueError(f"{sid}: 선형 강도는 start와 end만 쓴다 ({sorted(intensity)})")
        lo, hi = float(intensity["start"]), float(intensity["end"])
    else:
        lo = hi = float(intensity)
    return Cause(unit=unit, step=c["step"], location=str(c["location"]),
                 start_day=int(c["window"]["start_day"]), end_day=int(c["window"]["end_day"]),
                 pattern=c["pattern"], intensity_start=lo, intensity_end=hi)


def _validate(sid: str, c: Cause, fab: FabConfig) -> None:
    where = f"{sid} 원인 {c.location}"
    step = next((s for s in fab.steps if s.name == c.step), None)
    if step is None:
        raise ValueError(f"{where}: 없는 단계 {c.step}")
    if c.unit == "chamber":
        if c.location not in fab.chamber_ids():
            raise ValueError(f"{where}: 가상 팹에 없는 챔버")
        if "/" not in c.location:
            raise ValueError(f"{where}: 챔버는 TOOL/CH 형식이어야 한다")
    elif "/" in c.location:
        raise ValueError(f"{where}: 설비 단위 원인에 챔버를 적었다")
    if c.tool not in step.tools:
        raise ValueError(f"{where}: {c.tool}은 {c.step} 단계 설비가 아님")
    if not 1 <= c.start_day <= c.end_day <= fab.days:
        raise ValueError(f"{where}: 기간 {c.start_day}~{c.end_day}일차가 범위를 벗어남")
    if c.pattern not in CLASSES or c.pattern == "none":
        raise ValueError(f"{where}: 알 수 없는 불량 패턴 {c.pattern}")
    for v in (c.intensity_start, c.intensity_end):
        if not 0 <= v <= 1:
            raise ValueError(f"{where}: 강도는 0 이상 1 이하")
    if max(c.intensity_start, c.intensity_end) == 0:
        raise ValueError(f"{where}: 강도가 기간 내내 0이다")
