"""Bot ngẫu nhiên: chọn đều trong các hành động hợp lệ."""
from __future__ import annotations

import numpy as np

from autochess.core.actions import END_TURN


def act(view, mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal)) if legal.size else END_TURN


def random_actions(mask: np.ndarray, rng: np.random.Generator, end_bias: float = 0.0) -> np.ndarray:
    """Vector hóa cho lô: mask[B, P, A] -> actions[B, P]. end_bias > 0 tăng xác suất kết thúc lượt."""
    g = rng.random(mask.shape)
    if end_bias:
        g[..., END_TURN] += end_bias
    return np.where(mask, g, -1.0).argmax(-1)
