"""분류기 학습 루프. 실행은 scripts/train_classifier.py가 담당한다.

- 불균형 대응(balance=True): 에폭마다 none을 none_cap장만 뽑고, 빈도 역수 제곱근 가중치로 손실을 계산한다.
- 비교용(balance=False): 같은 장수를 train 몫에서 무작위로 뽑고 가중치 없이 학습한다.
  에폭당 연산량을 맞춰 불균형 대응의 효과만 비교하기 위해서다.
- 조기 종료: val 몫 macro-F1이 patience 에폭 동안 오르지 않으면 멈추고 최고 시점의 가중치를 쓴다.
"""

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from fabtrace.data.dataset import EpochSampler, WaferData, class_weights, encode, iter_batches
from fabtrace.data.wm811k import CLASSES
from fabtrace.models.cnn import WaferCNN


@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    none_cap: int = 10_000
    balance: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model: nn.Module, data: WaferData, indices: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        x = torch.from_numpy(encode(data.maps[indices[start:start + batch_size]]))
        out.append(torch.softmax(model(x), dim=1).numpy())
    return np.concatenate(out) if out else np.empty((0, len(CLASSES)), dtype=np.float32)


def macro_f1(model: nn.Module, data: WaferData, indices: np.ndarray) -> float:
    pred = predict_proba(model, data, indices).argmax(axis=1)
    return float(f1_score(data.labels[indices], pred, labels=range(len(CLASSES)), average="macro", zero_division=0))


def _epoch_indices(cfg: TrainConfig, sampler: EpochSampler, train: np.ndarray, epoch: int) -> np.ndarray:
    if cfg.balance:
        return sampler.epoch_indices(epoch)
    rng = np.random.default_rng([cfg.seed, epoch])
    return rng.choice(train, size=len(sampler), replace=False)


def fit(data: WaferData, cfg: TrainConfig, log=print) -> dict:
    """학습 후 최고 val macro-F1 시점의 모델과 기록을 담은 체크포인트 dict를 돌려준다."""
    set_seed(cfg.seed)
    train, val = data.indices("train"), data.indices("val")
    sampler = EpochSampler(data.labels, train, none_cap=cfg.none_cap, seed=cfg.seed)
    weights = torch.from_numpy(class_weights(data.labels[train], cfg.none_cap)) if cfg.balance else None

    model = WaferCNN()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history, best_f1, best_epoch, best_state, waited = [], -1.0, 0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        start = time.perf_counter()
        model.train()
        rng = np.random.default_rng([cfg.seed, epoch, 1])
        idx = _epoch_indices(cfg, sampler, train, epoch)
        total, seen = 0.0, 0
        for x, y in iter_batches(data, idx, cfg.batch_size, rng=rng):
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)

        val_f1 = macro_f1(model, data, val)
        row = {"epoch": epoch, "train_loss": total / seen, "val_macro_f1": val_f1,
               "seconds": time.perf_counter() - start}
        history.append(row)
        log(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val macro-F1 {val_f1:.4f}  {row['seconds']:.1f}s")

        if val_f1 > best_f1:
            best_f1, best_epoch, best_state, waited = val_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= cfg.patience:
                log(f"조기 종료: {cfg.patience}에폭 동안 개선 없음")
                break

    return {
        "state_dict": best_state,
        "classes": list(CLASSES),
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "torch_threads": torch.get_num_threads(),
    }


def load_model(path: str | Path) -> tuple[WaferCNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WaferCNN(n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt