"""Đối thủ cho agent và lịch chọn đối thủ (ưu tiên đối thủ khó).

Mọi đối thủ có cùng giao diện:
    act(state, mask[B, P, 44], seats[B]) -> actions[B]
state là State của một tập ván; mỗi ván có thể có ghế khác nhau.

OpponentPool giữ tỷ lệ thắng (trung bình trượt) của agent trước từng đối thủ và bốc đối thủ với
trọng số (1 - tỷ lệ thắng)^2, có sàn, để thời gian huấn luyện dồn vào đối thủ agent còn thua.
Với group_weights (giai đoạn 2), mỗi lần bốc trước hết chọn nhóm theo tỷ lệ cố định (vd bot luật 30%,
bản cũ của agent 70%), rồi mới ưu tiên đối thủ khó TRONG nhóm đó.
"""
from __future__ import annotations

import numpy as np

from autochess.core.actions import END_TURN
from autochess.bots.random_bot import random_actions
from autochess.bots.scripted import PRESETS, ScriptedBot


def _by_seat(fn, state, mask, seats):
    out = np.full(state.B, END_TURN, np.int64)
    for p in np.unique(seats):
        rows = np.flatnonzero(seats == p)
        sub = state.rows(rows) if rows.size < state.B else state
        out[rows] = fn(sub, mask[rows], int(p))
    return out


class BotOpponent:
    kind = "bot"

    def __init__(self, cat, name: str):
        self.name = name
        self.bot = ScriptedBot(cat, PRESETS[name])

    def act(self, state, mask, seats):
        return _by_seat(lambda s, m, p: self.bot.act(s, m, [p])[:, 0], state, mask, seats)


class RandomOpponent:
    kind = "bot"

    def __init__(self, seed: int = 0, end_bias: float = 0.1):
        self.name = "random"
        self.g = np.random.default_rng(seed)
        self.end_bias = end_bias

    def act(self, state, mask, seats):
        m = mask[np.arange(state.B), seats]                      # [B, 44]
        return random_actions(m[:, None, :], self.g, self.end_bias)[:, 0]


class PolicyOpponent:
    """Một bản mạng đóng băng (bản trong kho, hoặc agent khi đánh giá). policy: rl.policy.Policy.
    path: checkpoint của bản này (để chạy tiếp lần huấn luyện); protected: không bao giờ bị bỏ khỏi kho."""
    kind = "policy"

    def __init__(self, name: str, policy, cat, combat, matchup_row: bool = True, seed: int = 0,
                 path: str | None = None, protected: bool = False):
        self.name, self.policy, self.cat, self.combat = name, policy, cat, combat
        self.matchup_row = matchup_row
        self.g = np.random.default_rng(seed)
        self.path, self.protected = path, protected

    def act(self, state, mask, seats):
        from .observe import observe, flatten_batch
        obs = observe(self.cat, state, self.combat, self.matchup_row, seats=seats[:, None], mask=mask,
                      version=getattr(self.policy, "obs_version", 1))
        a, _, _ = self.policy.act(flatten_batch(obs), self.g)
        return a


def make_opponent(cat, combat, name: str, seed: int = 0):
    if name == "random":
        return RandomOpponent(seed)
    if name in PRESETS:
        return BotOpponent(cat, name)
    raise ValueError(f"không có đối thủ '{name}'")


class OpponentPool:
    def __init__(self, opponents: list, prioritize: bool = True, power: float = 2.0, floor: float = 0.05,
                 ema: float = 0.99, group_weights: dict | None = None):
        self.opponents = list(opponents)
        self.prioritize, self.power, self.floor, self.ema = prioritize, power, floor, ema
        self.group_weights = dict(group_weights) if group_weights else None
        self.winrate = np.full(len(self.opponents), 0.5)
        self.games = np.zeros(len(self.opponents), np.int64)

    @property
    def names(self):
        return [o.name for o in self.opponents]

    def add(self, opponent, winrate: float = 0.5):
        self.opponents.append(opponent)
        self.winrate = np.append(self.winrate, winrate)
        self.games = np.append(self.games, 0)

    def replace(self, i: int, opponent, winrate: float = 0.5):
        """Thay đối thủ ở ô i (giữ nguyên chỉ số để các ván đang chạy không bị lệch)."""
        self.opponents[i] = opponent
        self.winrate[i] = winrate
        self.games[i] = 0

    def kinds(self) -> np.ndarray:
        return np.array([getattr(o, "kind", "bot") for o in self.opponents])

    def _prio(self) -> np.ndarray:
        if not self.prioritize:
            return np.ones(len(self.opponents))
        return np.maximum((1.0 - self.winrate) ** self.power, self.floor)

    def weights(self) -> np.ndarray:
        """Xác suất bốc từng đối thủ (đã gộp tỷ lệ nhóm nếu có)."""
        w = self._prio()
        if self.group_weights:
            kinds = self.kinds()
            present = {g: gw for g, gw in self.group_weights.items() if (kinds == g).any() and gw > 0}
            total = sum(present.values())
            out = np.zeros_like(w)
            for g, gw in present.items():
                m = kinds == g
                out[m] = gw / total * w[m] / w[m].sum()
            w = out
        return w / w.sum()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self.opponents), size=n, p=self.weights())

    def update(self, idx: np.ndarray, score: np.ndarray) -> None:
        """score trong [0, 1]: 1 = agent thắng ván đó."""
        for i, s in zip(np.asarray(idx), np.asarray(score, np.float64)):
            self.winrate[i] = self.ema * self.winrate[i] + (1 - self.ema) * s
            self.games[i] += 1

    def state_dict(self) -> dict:
        return dict(names=self.names, winrate=self.winrate.tolist(), games=self.games.tolist())

    def load_state_dict(self, d: dict) -> None:
        for n, w, g in zip(d["names"], d["winrate"], d["games"]):
            if n in self.names:
                i = self.names.index(n)
                self.winrate[i], self.games[i] = w, g
