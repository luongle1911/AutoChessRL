"""Vòng lặp ván cho các ghế (B = 1). Runner không phân biệt người, bot hay phát lại."""
from __future__ import annotations

import numpy as np

from autochess.core import legal_actions, reset, step, check_invariants
from autochess.core.actions import END_TURN
from autochess.core.rng import make_seeds
from .adapter import to_view
from .seats import Command, Seat

LOG_FORMAT = "autochess-log-v1"


class GameRunner:
    def __init__(self, catalog, combat, seats: list[Seat], seed: int, debug: bool = True,
                 matchup_row: bool = True):
        self.catalog, self.combat, self.seats = catalog, combat, seats
        self.P = len(seats)
        self.seed = int(seed)
        self.debug = debug
        self.matchup_row = matchup_row
        for i, seat in enumerate(seats):
            if hasattr(seat, "attach"):
                seat.attach(self, i)
        self.reset()

    # ------------------------------------------------------------ trạng thái
    def reset(self) -> None:
        self.state = reset(self.catalog, make_seeds(self.seed, 1), self.P)
        self.history: list[list[int]] = []     # hành động từng bước
        self.human_flags: list[bool] = []      # bước đó có lệnh do người gõ không
        self.last_info = None

    @property
    def t(self) -> int:
        return len(self.history)

    @property
    def done(self) -> bool:
        return bool(self.state.done[0])

    def mask(self) -> np.ndarray:
        return legal_actions(self.catalog, self.state)[0]

    def view(self, seat: int, god: bool = False) -> dict:
        return to_view(self.catalog, self.state, 0, seat, god=god, combat=self.combat,
                       matchup_row=self.matchup_row)

    # ------------------------------------------------------------ chạy
    def collect(self):
        """Hỏi mọi ghế. Trả về (actions, human) hoặc một Command (khi đó chưa bước)."""
        m = self.mask()
        acts, human = [], False
        for i, seat in enumerate(self.seats):
            seat.on_step(self.t)
            legal = m[i]
            if not seat.always_ask and legal.sum() == 1 and legal[END_TURN]:
                acts.append(END_TURN)
                continue
            view = self.view(i) if seat.needs_view else None
            a = seat.act(None, legal, view)
            if isinstance(a, Command):
                return a
            if seat.kind == "human":
                human = True
            acts.append(int(a))
        return acts, human

    def advance(self, acts: list[int], human: bool = False) -> dict:
        if self.done:
            raise RuntimeError("ván đã kết thúc")
        self.state, info = step(self.catalog, self.combat, self.state, np.array([acts]), debug=self.debug)
        self.history.append(list(map(int, acts)))
        self.human_flags.append(human)
        self.last_info = info
        return info

    def tick(self):
        """Một bước: trả về info của step, hoặc Command nếu người gõ lệnh gỡ lỗi."""
        r = self.collect()
        if isinstance(r, Command):
            return r
        return self.advance(*r)

    def run_until(self, pred, max_steps: int = 100_000) -> dict | None:
        info = None
        for _ in range(max_steps):
            if self.done or pred(self):
                break
            r = self.tick()
            if isinstance(r, Command):
                return None
            info = r
        return info

    # ------------------------------------------------------------ undo / tái tạo
    def rebuild(self, history: list[list[int]], human_flags: list[bool] | None = None) -> None:
        """Chạy lại ván từ seed với chuỗi hành động cho trước (lõi tất định)."""
        self.reset()
        flags = human_flags or [False] * len(history)
        for acts, h in zip(history, flags):
            self.advance(acts, h)

    def undo(self) -> int:
        """Lùi về trước hành động gần nhất do người gõ. Trả về số bước đã bỏ (0 = không lùi được)."""
        if True not in self.human_flags:
            return 0
        k = len(self.human_flags) - 1 - self.human_flags[::-1].index(True)
        dropped = self.t - k
        self.rebuild(self.history[:k], self.human_flags[:k])
        return dropped

    def check(self) -> list[str]:
        return check_invariants(self.catalog, self.state)

    # ------------------------------------------------------------ log
    def to_log(self) -> dict:
        return dict(format=LOG_FORMAT, catalog_version=self.catalog.version,
                    combat_version=self.combat.version, seed=self.seed, num_players=self.P,
                    seats=[s.kind for s in self.seats], matchup_row=self.matchup_row,
                    actions=self.history, human_steps=[i for i, h in enumerate(self.human_flags) if h])
