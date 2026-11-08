"""Thu thập nhiều tiến trình (cho PC nhiều nhân): `workers` tiến trình con, mỗi tiến trình giữ một phần số ván
và một bản mạng trên CPU để chọn hành động; tiến trình chính chỉ học (trên GPU nếu có).

    python3 -m autochess.rl.train --config configs/p3_lobby.yaml configs/pc.yaml --set workers=16

Mỗi lần cập nhật:
  1. tiến trình chính gửi trọng số mạng (CPU), danh sách bản chụp trong kho, tỷ lệ thắng của kho;
  2. mỗi tiến trình con đồng bộ kho (nạp bản chụp mới từ đường dẫn), thu thập `steps` bước trên phần ván của
     nó, tính GAE, trải phẳng, đệm chiều đối thủ, rồi gửi lô về;
  3. tiến trình chính ghép các lô, ghi kết quả ván vào kho của nó (ưu tiên đối thủ khó tính ở đây).
Kết quả tất định theo seed với cùng số tiến trình (khác với chạy một tiến trình vì ván được chia khác).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import time

import numpy as np


def _make_pool(cat, combat, cfg):
    from .opponents import OpponentPool, make_opponent
    oc, lc = cfg["opponents"], cfg.get("league", {})
    use_league = bool(lc.get("seeds")) or int(lc.get("snapshot_every", 0)) > 0
    return OpponentPool([make_opponent(cat, combat, n, seed=cfg["seed"] + i) for i, n in enumerate(oc["names"])],
                        prioritize=oc["prioritize"], power=oc["power"], floor=oc["floor"],
                        group_weights=lc.get("group_weights") if use_league else None)


def _load_policy_opponent(cat, combat, e: dict, matchup_row: bool, seed: int):
    from .compat import load_compatible
    from .opponents import PolicyOpponent
    from .policy import Policy
    net, c, _ = load_compatible(e["path"], cat, "cpu", combat_version=combat.version)
    net.eval()
    return PolicyOpponent(e["name"], Policy(net), cat, combat, matchup_row=c.get("env", {}).get("matchup_row", matchup_row),
                          seed=seed, path=e["path"], protected=e.get("protected", False))


def _worker(conn, cfg: dict, wid: int):
    import torch
    torch.set_num_threads(1)
    from autochess.core import load_catalog, FormulaCombat
    from .nets import build_net
    from .policy import Policy
    from .train import collect_batch, make_rollouts
    cat = load_catalog(cfg.get("catalog"))
    combat = FormulaCombat(catalog=cat)
    pool = _make_pool(cat, combat, cfg)
    n_fixed = len(pool.opponents)
    net = build_net(cat, cfg["net"], cfg["num_players"])
    policy = Policy(net, "cpu")
    ro = None
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        _, sd, entries, winrate, games, update = msg
        # đồng bộ kho: các ô sau bot luật là bản neo / bản chụp, theo đúng số ô của tiến trình chính
        for e in sorted(entries, key=lambda e: e["slot"]):
            i = e["slot"]
            if i < len(pool.opponents) and pool.names[i] == e["name"]:
                continue
            opp = _load_policy_opponent(cat, combat, e, cfg["env"]["matchup_row"], cfg["seed"] + 101 * wid + i)
            if i < len(pool.opponents):
                pool.replace(i, opp)
            else:
                assert i == len(pool.opponents) and i >= n_fixed
                pool.add(opp)
        pool.winrate[:] = winrate
        pool.games[:] = games
        net.load_state_dict(sd)
        if ro is None:
            ro = make_rollouts(cat, combat, pool, cfg, update)
        batch, info = collect_batch(ro, policy, cfg["num_players"] - 1)
        conn.send((batch, info))


class VecRollout:
    """Thay cho Rollout / Lobbies trong train.py khi workers > 0."""

    def __init__(self, cfg: dict, workers: int, update: int = 0):
        ctx = mp.get_context("spawn")
        n = int(workers)
        total = int(cfg["env"]["num_games"])
        if total < n:
            raise ValueError(f"env.num_games ({total}) phải >= workers ({n})")
        shares = [total // n + (1 if k < total % n else 0) for k in range(n)]
        self.conns, self.procs = [], []
        for k, share in enumerate(shares):
            wcfg = copy.deepcopy(cfg)
            wcfg["env"]["num_games"] = share
            wcfg["seed"] = int(cfg["seed"]) + 7919 * (k + 1) + 1000 * update
            wcfg["device"], wcfg["threads"] = "cpu", 1
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, wcfg, k), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.total_steps = 0
        self.steps = int(cfg["env"]["steps"])

    def collect(self, net, pool, league=None, update: int = 0):
        """Trả về (batch, info) như train.collect_batch; kho `pool` của tiến trình chính được cập nhật."""
        t0 = time.perf_counter()
        sd = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in net.state_dict().items()}
        entries = league.state_dict() if league is not None else []
        for c in self.conns:
            c.send(("collect", sd, entries, pool.winrate.copy(), pool.games.copy(), update))
        results = [c.recv() for c in self.conns]
        batches = [r[0] for r in results]
        batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0] if k != "obs"}
        batch["obs"] = {k: np.concatenate([b["obs"][k] for b in batches]) for k in batches[0]["obs"]}
        finished = [f for r in results for f in r[1]["finished"]]
        idx = {n: i for i, n in enumerate(pool.names)}
        for f in finished:
            for a in f["agent_placement"]:
                for name, o in zip(f["opponents"], f["opp_placement"]):
                    if name in idx:
                        pool.update([idx[name]], [float(a < o)])
        self.total_steps += self.steps
        info = dict(seconds=time.perf_counter() - t0,
                    policy_seconds=float(np.mean([r[1]["policy_seconds"] for r in results])),
                    finished=finished)
        return batch, info

    def close(self):
        for c in self.conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
