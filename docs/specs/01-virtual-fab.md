# 01. 가상 팹 이력과 S1 결함 주입

- 마감: 2026-11-11
- 단계: phase-1
- 관련 설계: 설계서 2장(데이터 설계), 4.1(로트 이력 그래프)

## 목표

WM-811K 웨이퍼 맵을 합성 로트 이력에 배정해, 원인 챔버를 알고 있는 가상 팹 데이터를 만든다.
이후 모든 분석(03)의 정답지가 된다.

## 범위

- 공정 6단계 × 설비 2-3대 × 챔버 2-3개 구성
- 로트 25장 단위, 하루 8로트, 40일치 생성
- 웨이퍼별 단계마다 챔버 1개와 타임스탬프 기록
- S1 주입: `ETCH-03 / 챔버 B`, 12~18일차, 해당 챔버를 거친 웨이퍼의 Edge-Ring 비율 상승
- 함정: 모든 웨이퍼가 거치는 공통 병목 설비 1대
- 웨이퍼 맵 배정: 정상 구간은 `none` 위주, 주입 구간·챔버는 Edge-Ring 위주로 WM-811K 라벨 기준 샘플링

## 비범위

- SECOM 센서 데이터 연결
- Neo4j 적재 (이번에는 parquet + pandas로 처리)
- S2~S5 시나리오

## 산출물

| 경로 | 내용 |
| --- | --- |
| `configs/scenarios/s1.yaml` | 원인 위치, 기간, 패턴, 강도 |
| `src/fabtrace/sim/fab.py` | 공정 흐름과 챔버 배정 |
| `src/fabtrace/sim/generate.py` | 시나리오 실행 진입점 |
| `data/processed/genealogy.parquet` | wafer_id, lot_id, step, tool, chamber, ts |
| `data/processed/wafers.parquet` | wafer_id, map_index, true_pattern, day |
| `data/processed/ground_truth.json` | 주입한 원인 챔버와 기간 |

## 완료 기준

- [ ] `python -m fabtrace.sim.generate --scenario configs/scenarios/s1.yaml --seed 42` 실행 시 위 3개 파일이 생성된다
- [ ] 같은 시드로 두 번 실행한 결과가 내용이 같다 (`tests/sim/test_generate.py`)
- [ ] 주입 구간에서 `ETCH-03 / B` 경유 웨이퍼의 Edge-Ring 비율이 나머지보다 뚜렷이 높다 (`notebooks/01_sanity.ipynb` 그림 1장)
- [ ] 공통 병목 설비가 Edge-Ring 웨이퍼 경유 횟수 1위로 나온다 (함정이 실제로 작동하는지 확인)
- [ ] `data/`는 `.gitignore`에 들어 있고, 원데이터 다운로드 방법이 README에 있다

## 우선순위

| 순위 | 항목 |
| --- | --- |
| 필수 | S1 주입, 공통 병목 함정, 재현성 |
| 있으면 좋음 | 무관한 Random 불량 노이즈 |
| 미룸 | S3 시나리오 |

## 실험

### E1. 가상 팹 sanity 확인

- 관련 이슈: #5
- 노트북: `notebooks/01_sanity.ipynb`
- 시나리오와 시드: S1, 42

**가설**

1. 주입 기간(12~18일차)에 ETCH-03/B를 지난 웨이퍼의 Edge-Ring 비율이 다른 ETCH 챔버보다 뚜렷이 높다.
2. Edge-Ring 웨이퍼 경유 횟수로 챔버를 줄 세우면 1위는 공통 병목 CLN-01/A다.
3. 같은 순위에서 원인 챔버 ETCH-03/B는 상위권에 오르지 못한다. ETCH 챔버 하나가 처리하는 웨이퍼 비율이 다른 단계 챔버보다 작기 때문이다.

**방법**

- 그림 1: ETCH 통과 날짜 기준, 챔버별 하루 Edge-Ring 비율. 주입 기간을 음영으로 표시한다.
- 그림 2: 챔버별 Edge-Ring 웨이퍼 경유 횟수 순위. 원인 챔버와 병목 챔버를 색으로 구분한다.
- 시뮬레이션 자체를 검사하는 실험이므로 정답 컬럼 `true_pattern`을 사용한다.