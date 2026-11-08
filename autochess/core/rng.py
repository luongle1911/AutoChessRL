"""Số ngẫu nhiên không trạng thái.

Mỗi số ngẫu nhiên = băm SplitMix64 của bộ (seed ván, mục đích, các chỉ số vị trí).
Không có đối tượng sinh số nào cần lưu: seed[B] là đủ để tái tạo cả ván, và ván b
cho kết quả giống hệt dù chạy một mình hay trong lô.
"""
from __future__ import annotations

import numpy as np

_U64 = np.uint64
_GOLDEN = _U64(0x9E3779B97F4A7C15)
_M1 = _U64(0xBF58476D1CE4E5B9)
_M2 = _U64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = _U64(30), _U64(27), _U64(31), _U64(11)
_INV53 = 1.0 / float(1 << 53)

# Mục đích (tách luồng ngẫu nhiên cho từng việc)
ROLL_TIER = 1
ROLL_CHAMP = 2
PAIRING = 3
COMBAT = 4
RESEED = 5
BOT = 6

FREE_ROLL = 1000  # "chỉ số bước" dùng cho lần roll miễn phí đầu vòng (hành động thật < 1000)


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        return z ^ (z >> _S31)


def hash64(seed, *fields) -> np.ndarray:
    """Băm seed (mảng uint64) cùng các trường số nguyên không âm (vô hướng hoặc mảng, broadcast)."""
    h = np.asarray(seed, dtype=_U64)
    for f in fields:
        f = np.asarray(f)
        if f.dtype != _U64:
            f = f.astype(np.int64).astype(_U64)
        h = splitmix64(h ^ f)
    return h


def uniform(seed, *fields) -> np.ndarray:
    """Số thực đều trong [0, 1)."""
    return (hash64(seed, *fields) >> _S11).astype(np.float64) * _INV53


def make_seeds(base_seed: int, n: int) -> np.ndarray:
    """Sinh n seed ván từ một số nguyên."""
    return hash64(np.full(n, base_seed % (1 << 64), dtype=_U64), np.arange(n))
