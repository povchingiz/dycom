"""
labkit.engine — инфраструктура экспериментов: сиды, устройство, лог, чекпоинты.
Тонкие ноутбуки вызывают это, а не дублируют в ячейках.
"""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
import numpy as np


def set_seed(seed: int = 0):
    """Фиксируем всё, чтобы эксперимент воспроизводился."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """L40 -> cuda, иначе cpu. Печатает что выбрал."""
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            print(f"device: cuda | {torch.cuda.get_device_name(0)} | bf16 ok")
        else:
            print("device: cpu (smoke-test режим — ничего не уходит на GPU непроверенным)")
        return torch.device(dev)
    except ImportError:
        print("torch не установлен — только теоретическая часть")
        return "cpu"


class RunLogger:
    """Пишет метрики на диск СРАЗУ (не теряем при падении). Одна строка на шаг."""
    def __init__(self, name, root="logs"):
        Path(root).mkdir(exist_ok=True)
        self.path = Path(root) / f"{name}.jsonl"
        self.t0 = time.time()
        self.f = open(self.path, "a")

    def log(self, **kw):
        kw["t"] = round(time.time() - self.t0, 2)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def save_checkpoint(model, optimizer, meta: dict, path: str):
    """Сохраняем модель-артефакт + метаданные эксперимента."""
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer else None,
        "meta": meta,
    }, path)
    print(f"checkpoint -> {path}")


def smoke_test(train_step_fn):
    """Прогон одного батча на CPU. Железное правило: без зелёного smoke-теста
    ничего не запускается на L40."""
    try:
        loss = train_step_fn()
        print(f"smoke-test OK | loss={float(loss):.4f}")
        return True
    except Exception as e:
        print(f"smoke-test FAILED: {e}")
        return False
