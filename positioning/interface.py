"""Giao diện của agent xếp quân. Chưa có phần ruột.

Mọi mảng có chiều đầu là B (số trường hợp xử lý cùng lúc).
U = 20 ô tướng: chỉ số 0–9 là tướng trên sân, 10–19 là hàng chờ (đội mình);
với đối thủ, U ô là các tướng họ đã xếp lên sân.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PositioningInput:
    own_champ: np.ndarray       # (B, U) int8, -1 = trống
    own_star: np.ndarray        # (B, U) int8
    opp_champ: np.ndarray       # (B, K, U) int8
    opp_star: np.ndarray        # (B, K, U) int8
    opp_half_cell: np.ndarray   # (B, K, U) int8, ô nửa sân 0–27
    board_limit: np.ndarray     # (B,) int8, số tướng tối đa được lên sân (theo cấp)


@dataclass
class PositioningOutput:
    win_prob: np.ndarray        # (B, K) float32, xác suất thắng theo từng đối thủ
    half_cell: np.ndarray       # (B, U) int8, ô nửa sân của từng tướng, -1 = để hàng chờ


class PositioningAgent:
    """Lớp cơ sở. Agent vĩ mô chỉ gọi `evaluate`."""

    def evaluate(self, inp: PositioningInput) -> PositioningOutput:
        raise NotImplementedError
