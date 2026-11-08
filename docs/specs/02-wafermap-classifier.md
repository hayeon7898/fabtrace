# 02. 웨이퍼 맵 패턴 분류

- 마감: 2026-11-10
- 단계: phase-1
- 관련 설계: 설계서 5.1

## 목표

웨이퍼 맵만 보고 불량 패턴을 판정하는 분류기를 만든다.
03의 공통성 분석은 정답 라벨이 아니라 **이 모델의 예측값**을 입력으로 쓴다.

## 범위

- WM-811K 로드, 라벨 있는 맵만 학습에 사용
- 64×64 최근접 보간 리사이즈, 다이 값 0/1/2 유지
- 소형 CNN 1개 (ResNet 비교는 하지 않음)
- `none` 다운샘플링 + 클래스 가중치 + 회전·뒤집기 증강
- 01의 `wafers.parquet` 전체에 대한 예측 저장

## 비범위

- 임베딩 기반 유사 사례 검색
- 확률 보정(temperature scaling)
- 하이퍼파라미터 탐색

## 산출물

| 경로 | 내용 |
| --- | --- |
| `src/fabtrace/data/wm811k.py` | 로드와 전처리 |
| `src/fabtrace/models/cnn.py` | 모델 정의 |
| `scripts/train_classifier.py` | 학습 |
| `models/cnn_v1.pt` | 가중치 (git 제외) |
| `data/processed/predictions.parquet` | wafer_id, pred_pattern, confidence |
| `results/classifier_report.md` | 클래스별 precision, recall, F1과 혼동행렬 |

## 완료 기준

- [ ] `python scripts/train_classifier.py --seed 42` 로 학습과 평가가 한 번에 끝난다
- [ ] 테스트셋 macro-F1과 클래스별 recall이 `results/classifier_report.md`에 기록된다
- [ ] Edge-Ring recall이 보고서에 따로 적혀 있다 (03에서 가장 중요한 클래스)
- [ ] `predictions.parquet`가 `wafers.parquet`의 모든 wafer_id를 포함한다 (`pytest tests/test_predictions.py`)
- [ ] 정확도(accuracy)를 메인 지표로 쓰지 않는 이유가 보고서에 한 줄 있다

## 우선순위

| 순위 | 항목 |
| --- | --- |
| 필수 | 학습 스크립트, macro-F1, 예측 파일 |
| 있으면 좋음 | 혼동행렬 그림 |
| 미룸 | ResNet 비교, 임베딩 검색 |
