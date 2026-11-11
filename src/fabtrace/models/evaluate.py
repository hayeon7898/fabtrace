"""분류기 평가와 가상 팹 예측.

- 평가는 fab 몫 전체로 한다. fab 몫은 학습에도, 조기 종료(모델 선택)에도 쓰지 않았다.
- 03에 넘길 예측 모델은 val 몫 점수로 미리 고른다. fab 몫 결과를 보고 고르면 평가 숫자가 부풀려진다.
- 리포트는 숫자와 그림만 자동 생성한다. 해석은 실험 기록(E2)에 쓴다.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from fabtrace.data.dataset import WaferData  # noqa: E402
from fabtrace.data.wm811k import CLASSES  # noqa: E402
from fabtrace.models.train import load_model, predict_proba  # noqa: E402

LABELS = list(range(len(CLASSES)))


@dataclass
class Evaluation:
    name: str
    ckpt: dict
    per_class: pd.DataFrame  # index=CLASSES, columns=precision, recall, f1, support
    macro_f1: float
    accuracy: float
    confusion: np.ndarray


def evaluate(name: str, ckpt: dict, y_true: np.ndarray, y_pred: np.ndarray) -> Evaluation:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(CLASSES))
    return Evaluation(
        name=name,
        ckpt=ckpt,
        per_class=per_class,
        macro_f1=float(f.mean()),
        accuracy=float((y_true == y_pred).mean()),
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS),
    )


def evaluate_model(path: Path, data: WaferData, pool: str = "fab") -> Evaluation:
    model, ckpt = load_model(path)
    idx = data.indices(pool)
    pred = predict_proba(model, data, idx).argmax(axis=1)
    return evaluate(Path(path).stem, ckpt, data.labels[idx], pred)


def predict_wafers(path: Path, data: WaferData, wafers: pd.DataFrame) -> pd.DataFrame:
    """가상 팹 웨이퍼마다 예측 패턴, 신뢰도, 9개 클래스 확률."""
    model, _ = load_model(path)
    probs = predict_proba(model, data, wafers["map_index"].to_numpy())
    out = pd.DataFrame({
        "wafer_id": wafers["wafer_id"].to_numpy(),
        "pred_pattern": np.array(CLASSES)[probs.argmax(axis=1)],
        "confidence": probs.max(axis=1),
    })
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i]
    return out


def plot_confusion(ev: Evaluation, path: Path) -> None:
    cm = ev.confusion / ev.confusion.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(LABELS, CLASSES, rotation=45, ha="right")
    ax.set_yticks(LABELS, CLASSES)
    for i in LABELS:
        for j in LABELS:
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{ev.name}: confusion matrix (row-normalized = recall)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(evals: list[Evaluation], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for ev in evals:
        h = pd.DataFrame(ev.ckpt["history"])
        ax.plot(h["epoch"], h["val_macro_f1"], marker="o", ms=3, label=ev.name)
        ax.scatter([ev.ckpt["best_epoch"]], [ev.ckpt["best_val_macro_f1"]], s=60, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Validation macro-F1 by epoch (dot = selected epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def write_report(evals: list[Evaluation], predicted_with: str | None, n_predicted: int,
                 figures: dict[str, str], n_eval: int, path: Path) -> None:
    none_share = evals[0].per_class.loc["none", "support"] / evals[0].per_class["support"].sum()
    lines = [
        "# 웨이퍼 맵 분류기 평가 리포트",
        "",
        "> `scripts/evaluate_classifier.py`가 자동 생성한다. 직접 고치지 말 것. 해석은 실험 기록 E2에 있다.",
        "",
        f"평가 데이터는 fab 몫 {n_eval:,}장이다. 학습에도, 조기 종료에도 쓰지 않았다.",
        "모델 선택은 val 몫 macro-F1로 했고, fab 몫은 선택 후 확인에만 썼다.",
        "",
        "## 모델 요약",
        "",
        "| 모델 | 불균형 대응 | 선택 에폭 | val macro-F1 | fab macro-F1 | fab Edge-Ring recall | fab 정확도 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ev in evals:
        lines.append(
            f"| `{ev.name}` | {'O' if ev.ckpt['config']['balance'] else 'X'} | {ev.ckpt['best_epoch']} "
            f"| {_fmt(ev.ckpt['best_val_macro_f1'])} | {_fmt(ev.macro_f1)} "
            f"| {_fmt(ev.per_class.loc['Edge-Ring', 'recall'])} | {_fmt(ev.accuracy)} |"
        )
    lines += [
        "",
        f"정확도는 메인 지표로 쓰지 않는다. 평가 데이터의 {none_share:.0%}가 none이라 none만 맞혀도 정확도가 높게 나온다.",
        "",
        "## 클래스별 recall 비교",
        "",
        "| 클래스 | 평가 장수 | " + " | ".join(f"`{ev.name}`" for ev in evals) + " |",
        "| --- | --- | " + " | ".join("---" for _ in evals) + " |",
    ]
    for c in CLASSES:
        lines.append(f"| {c} | {int(evals[0].per_class.loc[c, 'support']):,} | "
                     + " | ".join(_fmt(ev.per_class.loc[c, "recall"]) for ev in evals) + " |")

    for ev in evals:
        lines += ["", f"## `{ev.name}` 클래스별 지표", "",
                  "| 클래스 | precision | recall | F1 |", "| --- | --- | --- | --- |"]
        for c in CLASSES:
            row = ev.per_class.loc[c]
            lines.append(f"| {c} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} |")

    lines += ["", "## 그림", ""]
    lines += [f"![{title}]({rel})" for title, rel in figures.items()]
    if predicted_with:
        lines += ["", "## 가상 팹 예측", "",
                  f"`{predicted_with}`로 가상 팹 웨이퍼 {n_predicted:,}장을 예측해 "
                  "`data/processed/predictions.parquet`에 저장했다. 예측 패턴, 신뢰도, 9개 클래스 확률을 담는다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(model_paths: list[Path], data: WaferData, wafers: pd.DataFrame | None,
                   predict_with: Path | None, results_dir: Path, processed_dir: Path) -> list[Evaluation]:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    evals = [evaluate_model(p, data) for p in model_paths]

    figures = {}
    for ev in evals:
        plot_confusion(ev, fig_dir / f"confusion_{ev.name}.png")
        figures[f"{ev.name} 혼동행렬"] = f"figures/confusion_{ev.name}.png"
    plot_learning_curves(evals, fig_dir / "classifier_learning_curves.png")
    figures["val macro-F1 학습 곡선"] = "figures/classifier_learning_curves.png"

    n_predicted = 0
    if predict_with is not None and wafers is not None:
        preds = predict_wafers(predict_with, data, wafers)
        processed_dir.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(processed_dir / "predictions.parquet", index=False)
        n_predicted = len(preds)

    write_report(evals, Path(predict_with).stem if predict_with else None, n_predicted,
                 figures, len(data.indices("fab")), results_dir / "classifier_report.md")
    return evals