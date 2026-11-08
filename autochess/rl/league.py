"""Kho phiên bản cho giai đoạn 2 (tự đấu): các bản chụp đóng băng của agent làm đối thủ.

    league = League(pool, cat, combat, run_dir, cfg["league"])
    league.add_seed("runs/.../update_00250.pt", "v2_250")    # bản neo, không bao giờ bị bỏ
    ...
    league.maybe_snapshot(net, cfg, update)                   # sau mỗi lần cập nhật

Mỗi `snapshot_every` lần cập nhật, mạng hiện tại được lưu vào run_dir/snapshots/ và thêm vào kho đối thủ
(OpponentPool, nhóm "policy"). Kho có tối đa `max_snapshots` bản chụp (không tính bản neo); khi đầy, bản
mới thế chỗ bản mà agent đang thắng nhiều nhất (đối thủ ít giá trị học nhất) trong số các bản đã đấu ít nhất
`min_games` ván; nếu chưa bản nào đủ ván thì thế chỗ bản cũ nhất. Bản neo (seeds) luôn giữ.

Tỷ lệ chọn đối thủ đặt ở chỗ khác: tự đấu = opponents.selfplay_prob (Rollout), còn lại chia theo
league.group_weights giữa bot luật ("bot") và bản cũ ("policy").
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .compat import load_compatible
from .opponents import PolicyOpponent
from .policy import Policy


class League:
    def __init__(self, pool, cat, combat, run_dir: str, cfg: dict, matchup_row: bool = True, seed: int = 0):
        self.pool, self.cat, self.combat = pool, cat, combat
        self.dir = os.path.join(run_dir, "snapshots")
        os.makedirs(self.dir, exist_ok=True)
        self.every = int(cfg.get("snapshot_every", 25))
        self.max_snapshots = int(cfg.get("max_snapshots", 10))
        self.min_games = int(cfg.get("min_games", 100))
        self.matchup_row = matchup_row
        self.seed = seed

    # ------------------------------------------------------------ nạp
    def _load(self, path: str, name: str, protected: bool) -> PolicyOpponent:
        net, cfg, rep = load_compatible(path, self.cat, "cpu", combat_version=self.combat.version)
        net.eval()
        mr = cfg.get("env", {}).get("matchup_row", self.matchup_row)
        return PolicyOpponent(name, Policy(net), self.cat, self.combat, matchup_row=mr,
                              seed=self.seed + len(self.pool.opponents), path=path, protected=protected)

    def add_seed(self, path: str, name: str | None = None) -> None:
        name = name or os.path.splitext(os.path.basename(path))[0]
        if name in self.pool.names:
            return
        self.pool.add(self._load(path, name, protected=True))

    def snapshots(self) -> list[int]:
        """Chỉ số trong pool của các bản chụp có thể bị thay (không phải bản neo)."""
        return [i for i, o in enumerate(self.pool.opponents)
                if getattr(o, "kind", "") == "policy" and not getattr(o, "protected", False)]

    # ------------------------------------------------------------ chụp
    def maybe_snapshot(self, net, cfg: dict, update: int) -> str | None:
        if self.every <= 0 or update % self.every:
            return None
        name = f"snap_{update:05d}"
        path = os.path.join(self.dir, f"{name}.pt")
        torch.save(dict(net={k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in net.state_dict().items()},
                        config=cfg, update=update), path)
        opp = self._load(path, name, protected=False)
        snaps = self.snapshots()
        if len(snaps) < self.max_snapshots:
            self.pool.add(opp)
            return f"thêm {name} vào kho ({len(snaps) + 1}/{self.max_snapshots} bản chụp)"
        # đầy: thay bản agent đang thắng nhiều nhất (học được ít nhất từ nó), chỉ xét bản đã đủ ván;
        # chưa bản nào đủ ván thì thay bản cũ nhất
        ready = [j for j in snaps if self.pool.games[j] >= self.min_games]
        if ready:
            i = max(ready, key=lambda j: self.pool.winrate[j])
        else:
            i = min(snaps, key=lambda j: self.pool.names[j])
        old = self.pool.names[i]
        self.pool.replace(i, opp)
        return f"thêm {name} vào kho, thay {old}"

    # ------------------------------------------------------------ chạy tiếp
    def state_dict(self) -> list[dict]:
        return [dict(name=o.name, path=o.path, protected=o.protected, slot=i)
                for i, o in enumerate(self.pool.opponents) if getattr(o, "kind", "") == "policy"]

    def load_state_dict(self, entries: list[dict]) -> None:
        """Dựng lại các đối thủ dạng mạng đúng ô cũ (bot luật đã có sẵn ở các ô đầu)."""
        for e in sorted(entries, key=lambda e: e["slot"]):
            if not e.get("path") or not os.path.exists(e["path"]):
                print(f"bỏ qua {e['name']}: không thấy {e.get('path')}")
                continue
            opp = self._load(e["path"], e["name"], e.get("protected", False))
            if e["slot"] < len(self.pool.opponents):
                self.pool.replace(e["slot"], opp)
            else:
                self.pool.add(opp)

    def summary(self) -> str:
        parts = []
        for i, o in enumerate(self.pool.opponents):
            if getattr(o, "kind", "") == "policy":
                parts.append(f"{o.name}{'*' if o.protected else ''}:{100 * self.pool.winrate[i]:.0f}")
        return " ".join(parts)
