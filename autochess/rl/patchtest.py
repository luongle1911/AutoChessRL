"""Đo một agent trên một file dữ liệu (dùng cho thử "bản cập nhật luật").

    python3 -m autochess.rl.patchtest --ckpt runs/.../latest.pt --catalog autochess/data/catalog_v1b.json \\
        --vs runs/p2_bots_v2_20261103_191704/checkpoints/update_00250.pt --out ketqua.json

In ra: báo cáo ghép tên (tướng/tộc giữ, mới, bỏ), tỷ lệ thắng trước các bot luật trên luật đó, đấu trực tiếp
với các checkpoint --vs, và tỷ lệ mua từng tướng khi tướng đó xuất hiện trong shop mà đủ vàng
(để xem agent có dùng tướng mới, tướng vừa được tăng sức mạnh hay không).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from autochess import tournament as T
from autochess.core import load_catalog, FormulaCombat
from autochess.core.actions import BUY0, SELL0
from autochess.core.combat import board_features


class TraitShare:
    """Bọc module giao tranh: ghi phần sức mạnh đến từ tộc (thưởng / (1 + thưởng)) của một ghế mỗi trận."""

    def __init__(self, inner):
        self.inner, self.version = inner, inner.version
        self.seat, self.vals = 0, []

    def evaluate(self, cat, state):
        f = board_features(cat, state)
        b = f["bonus"][:, self.seat]
        self.vals.append((b / (1 + b))[f["n_board"][:, self.seat] > 0])
        return self.inner.evaluate(cat, state)


class BuyCounter:
    """Bọc AgentPlayer: đếm số lần mỗi tướng được chào bán (đủ vàng) và được mua."""

    def __init__(self, agent, N: int):
        self.agent = agent
        self.offers = np.zeros(N, np.int64)
        self.buys = np.zeros(N, np.int64)

    def act(self, state, mask, seats):
        a = self.agent.act(state, mask, seats)
        b = np.arange(state.B)[:, None]
        seats_arr = np.tile(np.atleast_1d(np.asarray(seats)), (state.B, 1))
        live = state.alive[b, seats_arr] & ~state.done[:, None] & ~state.ended_turn[b, seats_arr]
        dec = live & (mask[b, seats_arr].sum(-1) > 1)
        shop = state.shop[b, seats_arr].astype(np.int64)                     # [B,S,5]
        gold = state.gold[b, seats_arr]
        cost = np.where(shop >= 0, self.agent.cat.cost[np.clip(shop, 0, None)], 99)
        offered = dec[..., None] & (shop >= 0) & (gold[..., None] >= cost)
        np.add.at(self.offers, shop[offered], 1)
        bought = dec & (a >= BUY0) & (a < SELL0)
        if bought.any():
            bi, si = np.nonzero(bought)
            np.add.at(self.buys, shop[bi, si, a[bi, si] - BUY0], 1)
        return a


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--vs", nargs="*", default=[])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--h2h-games", type=int, default=1024)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from .agent import AgentPlayer
    from .compat import load_compatible
    from .evaluate import evaluate, format_eval, head_to_head
    from .policy import Policy
    cat = load_catalog(a.catalog)
    combat = FormulaCombat(catalog=cat)
    net, cfg, rep = load_compatible(a.ckpt, cat, combat_version=combat.version)
    print(rep)
    pol = Policy(net)
    out = dict(ckpt=a.ckpt, catalog=cat.version, report=str(rep))
    bots = ("tempo", "hyper", "roll7", "fast9", "bank", "random")
    res = evaluate(cat, combat, pol, opponents=bots, games=a.games, B=a.B, matchup_row=cfg["env"]["matchup_row"])
    print(format_eval(res))
    out["eval"] = res
    out["h2h"] = {}
    for p in a.vs:
        nb, cb, _ = load_compatible(p, cat)
        r = head_to_head(cat, combat, pol, Policy(nb), games=a.h2h_games, B=256, seed=2026,
                         matchup_row=cfg["env"]["matchup_row"], matchup_row_b=cb["env"]["matchup_row"])
        print(f"đấu trực tiếp {p}: thắng {100 * r['winrate_a']:.1f}% ± {100 * r['stderr']:.1f}")
        out["h2h"][p] = r
    # tỷ lệ mua từng tướng
    counter = BuyCounter(AgentPlayer(pol, cat, combat, matchup_row=cfg["env"]["matchup_row"], seed=9), cat.N)
    ts = TraitShare(combat)
    for bot in bots[:5]:
        for lineup in (("agent", bot), (bot, "agent")):
            ts.seat = lineup.index("agent")
            T.play(cat, ts, lineup, 100, seed=31, B=a.B, players={"agent": counter})
    out["trait_share"] = float(np.concatenate(ts.vals).mean())
    print(f"phần sức mạnh của agent đến từ tộc: {100 * out['trait_share']:.0f}%")
    rate = counter.buys / np.maximum(counter.offers, 1)
    out["buy_rate"] = {n: dict(cost=int(c), offers=int(o), buys=int(b), rate=float(r)) for n, c, o, b, r in
                       zip(cat.names, cat.cost, counter.offers, counter.buys, rate)}
    by_cost = {t: [n for n in cat.names if out["buy_rate"][n]["cost"] == t] for t in range(1, 6)}
    out["buys_by_cost"] = {t: int(sum(out["buy_rate"][n]["buys"] for n in ns)) for t, ns in by_cost.items()}
    print("số lần mua theo giá: " + ", ".join(f"{t} tiền {v}" for t, v in out["buys_by_cost"].items()))
    print("tỷ lệ mua khi được chào (đủ vàng):")
    for t in range(1, 6):
        row = [f"{n} {100 * out['buy_rate'][n]['rate']:.0f}%" for n in cat.names if out["buy_rate"][n]["cost"] == t]
        print(f"  {t} tiền: " + ", ".join(row))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
