# 03. 공통성 분석과 격리 범위 산정

- 마감: 2026-11-14
- 단계: phase-1
- 관련 설계: 설계서 4.2, 7장, 11장

## 목표

Edge-Ring 클레임이 들어왔다고 가정하고, 원인 챔버를 찾고 출하 보류 범위를 정한다.
단순 빈도 집계는 함정에 걸리고 통계 방법은 맞힌다는 것을 결과로 보인다.

## 범위

- 입력: 01의 이력 + 02의 예측 (정답 라벨 사용 금지)
- 방법 A: 챔버별 Edge-Ring 웨이퍼 경유 횟수 순위
- 방법 B: 챔버별 lift + 피셔 정확 검정 + Benjamini–Hochberg 보정
- 격리: 원인 챔버를 주입 기간에 거친 모든 로트를 순방향 추적
- 효과 확인: 패턴 비율 p 관리도, 조치 전후 구간 비교

## 비범위

- 센서 교차 확인
- LLM 에이전트
- 여러 시나리오 반복 평가 (S1 하나, 시드 10개까지만)

## 산출물

| 경로 | 내용 |
| --- | --- |
| `src/fabtrace/analysis/spc.py` | p 관리도, 이상 구간 탐지 |
| `src/fabtrace/analysis/commonality.py` | 방법 A, B |
| `src/fabtrace/analysis/containment.py` | 노출 기간 추정, 순방향 추적, 채점 |
| `src/fabtrace/analysis/pipeline.py` | 감시, 공통성, 격리를 엮는 분석과 채점, 가상 팹 생성과 예측 |
| `src/fabtrace/analysis/report.py` | 리포트, JSON, 그림 저장 |
| `scripts/run_rca.py` | 분석 전체 실행 |
| `results/reports/s1_rca.md` | 감시 결과, 방법 비교 표, 격리 로트 목록, 채점 |
| `results/reports/s1_rca.json` | 분석 결과(`analysis`)와 채점(`evaluation`). 스펙 05 입력은 `analysis`만 |
| `results/figures/s1_rca_*.png` | p 관리도, 공통성 비교 막대 |

## 완료 기준

- [ ] `python scripts/run_rca.py --scenario s1 --seeds 0-9` 실행 시 `results/s1_rca.md`가 생성된다
- [ ] 방법 A의 Top-1이 공통 병목 설비이고, 방법 B의 Top-1이 `ETCH-03 / B`인 시드 비율이 표로 나온다
- [ ] 방법 B 결과에 lift, p-value, q-value가 함께 있다
- [ ] 격리 로트 목록과 실제 영향 로트(ground_truth 기준)의 precision, recall이 기록된다
- [ ] 주입 구간에서 관리한계를 벗어나고 이후 돌아오는 p 관리도가 그려진다
- [ ] 분석 코드가 `true_pattern` 컬럼을 읽지 않는다 (`pytest tests/test_no_leak.py`)

## 우선순위

| 순위 | 항목 |
| --- | --- |
| 필수 | 방법 A·B 비교, 격리 목록 |
| 있으면 좋음 | p 관리도, 시드 10개 반복 |
| 미룸 | S3 시나리오 추가 |


## 실험
 
### E3. 시드 10개 반복 평가
 
- 관련 이슈: #26
- 스크립트: `scripts/run_e3.py`
- 시나리오와 시드: S1, 평가 시드 0~9 (개발용 seed 42는 제외하고 참고로만 표시)

**가설**
 
1. 방법 B는 시드 10개 모두에서 원인 챔버 ETCH-03/B를 1위로 고르고, 방법 A는 병목 CLN-01/A를 1위로 고른다.
2. 여유 24시간 격리는 시드 10개 모두에서 영향 로트를 놓치지 않는다.
3. 일부 시드에서는 rule1만으로 이상 구간을 놓치고, rule2가 이를 보완한다.
4. 다른 패턴 오탐과 잘못 지목한 원인 챔버는 거의 없다.

**방법**
 
- 시드마다 가상 팹(이력, 맵 배정)을 새로 만들고 `cnn_v1`로 예측한다. 분류기는 다시 학습하지 않는다.
- 시드마다 `run_analysis`로 감시, 공통성, 격리를 실행하고 정답지로 채점한다.
- 격리는 여유 0, 12, 24시간을 모두 계산하고, 감지는 rule1만 쓸 때와 rule2를 더할 때를 비교한다.
- 적중률은 10개 중 몇 개, precision과 recall은 평균과 최솟값으로 요약한다.
