"""Thu thập dữ liệu (quan sát, hành động) từ các ván bot luật đấu nhau, để học bắt chước (rl.bc).

Mỗi ván chọn ngẫu nhiên một bot cho mỗi ghế (có thể trùng nhau); hết ván thì chọn lại. Chỉ giữ các bước
"quyết định thật" (ghế còn trong ván và có hơn một hành động hợp lệ). Không cần PyTorch.
"""
from __future__ import annotations

import numpy as np

from autochess.core import reset, step, legal_actions, rng as crng
from autochess.core.actions import END_TURN
from .observe import observe
from .opponents import BotOpponent


class BotDataCollector:
    def __init__(self, cat, combat, bots=("tempo", "hyper", "roll7", "fast9"), num_games: int = 256,
                 num_players: int = 2, seed: int = 0, matchup_row: bool = True, obs_version: int = 1):
        self.cat, self.combat = cat, combat
        self.bots = [BotOpponent(cat, n) for n in bots]
        self.B, self.P = num_games, num_players
        self.matchup_row = matchup_row
        self.obs_version = obs_version
        self.rng = np.random.default_rng(seed)
        self.state = reset(cat, crng.make_seeds(seed, num_games), num_players)
        self.owner = self.rng.integers(len(self.bots), size=(num_games, num_players))
        self.was_done = np.zeros(num_games, bool)

    def collect(self, steps: int) -> dict:
        """Chạy `steps` bước, trả về {obs: {khóa: [M, ...]}, actions: [M], bot: [M]}."""
        parts, acts, who = [], [], []
        B, P = self.B, self.P
        for _ in range(steps):
            s = self.state
            m = legal_actions(self.cat, s)
            actions = np.full((B, P), END_TURN, np.int64)
            for k, bot in enumerate(self.bots):
                rb, ps = np.nonzero(self.owner == k)
                if rb.size:
                    actions[rb, ps] = bot.act(s.rows(rb), m[rb], ps)
            active = s.alive & ~s.done[:, None]
            dec = active & (m.sum(-1) > 1)
            if dec.any():
                obs = observe(self.cat, s, self.combat, self.matchup_row, mask=m, version=self.obs_version)
                bi, pi = np.nonzero(dec)
                parts.append({k: v[bi, pi] for k, v in obs.items()})
                acts.append(actions[bi, pi])
                who.append(self.owner[bi, pi])
            self.state, _ = step(self.cat, self.combat, s, actions, mask=m)
            # ván vừa kết thúc: chọn bot mới cho ván kế tiếp
            new = self.state.done & ~self.was_done
            if new.any():
                self.owner[new] = self.rng.integers(len(self.bots), size=(int(new.sum()), P))
            self.was_done = self.state.done.copy()
        if not parts:
            return dict(obs={}, actions=np.zeros(0, np.int64), bot=np.zeros(0, np.int64))
        obs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        return dict(obs=obs, actions=np.concatenate(acts), bot=np.concatenate(who))
