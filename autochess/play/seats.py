"""Ghế ngồi. Mọi loại ghế có cùng một hàm act(obs, mask, view) -> int.

HumanSeat    hiển thị view và chờ lệnh (có thể trả về Command gỡ lỗi thay cho số hiệu)
ScriptedSeat bot luật, đọc view
ArrayBotSeat bot luật vector hóa (bots/scripted.py), đọc thẳng mảng trạng thái của runner
PolicySeat   bot mạng, đọc obs (observe.py làm ở bước sau)
ReplaySeat   đọc số hiệu từ log
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autochess.core.actions import END_TURN


@dataclass
class Command:
    """Lệnh gỡ lỗi (không phải hành động, không vào log)."""
    name: str
    arg: str = ""


class Seat:
    kind = "seat"
    needs_view = True     # runner chỉ dựng view khi ghế cần
    always_ask = False    # True: hỏi cả khi chỉ còn hành động 43

    def on_step(self, t: int) -> None:
        pass

    def act(self, obs, mask: np.ndarray, view: dict | None):
        raise NotImplementedError


class HumanSeat(Seat):
    kind = "human"

    def __init__(self, input_fn):
        self.input_fn = input_fn   # input_fn(view, mask) -> int | Command

    def act(self, obs, mask, view):
        return self.input_fn(view, mask)


class ScriptedSeat(Seat):
    def __init__(self, policy_fn, name: str, seed: int = 0):
        self.policy_fn = policy_fn  # policy_fn(view, mask, rng) -> int
        self.kind = f"bot:{name}"
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, view):
        a = int(self.policy_fn(view, mask, self.rng))
        return a if mask[a] else END_TURN


class ArrayBotSeat(Seat):
    """Bot luật vector hóa. Runner gọi attach(runner, seat) để bot đọc trạng thái hiện tại."""
    needs_view = False

    def __init__(self, bot, name: str):
        self.bot = bot
        self.kind = f"bot:{name}"
        self.runner, self.seat = None, None

    def attach(self, runner, seat: int) -> None:
        self.runner, self.seat = runner, seat

    def act(self, obs, mask, view):
        a = self.bot.act_one(self.runner.state, mask, 0, self.seat)
        return a if mask[a] else END_TURN


class PolicySeat(Seat):
    kind = "policy"
    needs_view = False

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn  # policy_fn(obs, mask) -> int

    def act(self, obs, mask, view):
        if obs is None:
            raise RuntimeError("PolicySeat cần obs; observe.py chưa được nối vào runner")
        return int(self.policy_fn(obs, mask))


class ReplaySeat(Seat):
    kind = "replay"
    needs_view = False
    always_ask = True

    def __init__(self, actions: list):
        self.actions = actions
        self.t = 0

    def on_step(self, t: int) -> None:
        self.t = t

    def act(self, obs, mask, view):
        if self.t >= len(self.actions):
            raise StopIteration("log đã hết hành động")
        return int(self.actions[self.t])
