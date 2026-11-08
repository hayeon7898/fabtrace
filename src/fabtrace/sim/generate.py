"""가상 팹 데이터 생성의 유일한 진입점.

공정 이력 생성과 결함 주입을 같은 시드로 한 번에 실행해, 파일끼리 어긋나는 일을 막는다.

사용법: uv run python -m fabtrace.sim.generate --scenario configs/scenarios/s1.yaml --seed 42

출력 (data/processed/)
  genealogy.parquet   웨이퍼별 단계, 설비, 챔버, 시각
  wafers.parquet      웨이퍼별 배정 맵, 실제 패턴, 영향 여부
  ground_truth.json   주입한 원인과 영향 로트 목록
"""

import argparse
import json
from pathlib import Path

from fabtrace.data.wm811k import assign_pools, load_meta
from fabtrace.sim.config import load_fab_config
from fabtrace.sim.genealogy import generate_genealogy
from fabtrace.sim.inject import build_wafers, ground_truth
from fabtrace.sim.scenario import load_scenario


def run(fab_path: Path, scenario_path: Path, seed: int, data_dir: Path, out_dir: Path) -> dict:
    fab = load_fab_config(fab_path)
    sc = load_scenario(scenario_path, fab)
    meta = load_meta(data_dir)
    pool = meta[assign_pools(meta["label"]) == "fab"]

    genealogy = generate_genealogy(fab, seed)
    wafers = build_wafers(genealogy, sc, fab, pool, seed)
    gt = ground_truth(sc, fab, wafers, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    genealogy.to_parquet(out_dir / "genealogy.parquet", index=False)
    wafers.to_parquet(out_dir / "wafers.parquet", index=False)
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="가상 팹 이력 생성과 결함 주입")
    parser.add_argument("--scenario", type=Path, default=Path("configs/scenarios/s1.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fab", type=Path, default=Path("configs/fab.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    gt = run(args.fab, args.scenario, args.seed, args.data, args.out)
    print(f"{gt['scenario']} (seed {gt['seed']}): 원인 {gt['chamber']}, "
          f"영향 {gt['n_affected_wafers']}장 / {len(gt['affected_lots'])}로트, "
          f"주입 {gt['n_injected_wafers']}장 -> {args.out}/")


if __name__ == "__main__":
    main()