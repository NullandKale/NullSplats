"""Registry of optimizers and LR schedules -- "method" hyperparameters for how training itself
proceeds, independent of model architecture. Named optim.py (not to be confused with torch.optim,
which every caller here still accesses via the fully-qualified `torch.optim.X`).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

OPTIMIZERS: dict[str, Callable[[Iterable[torch.nn.Parameter], float], torch.optim.Optimizer]] = {
    "adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
    "adamw": lambda params, lr: torch.optim.AdamW(params, lr=lr),
    "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
}


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    key = name.strip().lower()
    if key not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer: {name!r}. Available: {list_optimizers()}")
    return OPTIMIZERS[key](params, lr)


def list_optimizers() -> list[str]:
    return sorted(OPTIMIZERS)


SCHEDULERS = ("constant", "cosine", "step")


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, total_steps: int):
    """total_steps = number of times .step() will be called on this scheduler (one call per
    epoch). Returns None for 'constant' -- callers should skip stepping in that case."""
    key = name.strip().lower()
    if key == "constant":
        return None
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if key == "step":
        step_size = max(1, total_steps // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    raise KeyError(f"Unknown lr_schedule: {name!r}. Available: {list_schedulers()}")


def list_schedulers() -> list[str]:
    return list(SCHEDULERS)
