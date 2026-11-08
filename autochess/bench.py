"""Đo tốc độ lõi: bước-ván mỗi giây với B = 1, 64, 1024.

    python3 -m autochess.bench [--players 2] [--steps 300] [--profile]
    python3 -m autochess.bench --bots tempo,hyper,roll7,fast9 --players 8   # bot luật, sát thực tế hơn

Bot ngẫu nhiên kết thúc lượt rất sớm nên giao tranh dày hơn thực tế; bot luật cho số đo gần
với lúc huấn luyện (agent làm 5-15 hành động mỗi vòng). "quyết định/giây" = bước-ván x P.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from autochess.core import load_catalog, FormulaCombat, reset, step, legal_actions, rng
from autochess.core.actions import END_TURN
from autochess.bots.random_bot import random_actions
from autochess.bots.scripted import PRESETS, ScriptedBot


def run(cat, combat, B, P, steps, seed=0, bots=None):
    """bots: None = bot ngẫu nhiên; hoặc danh sách tên bot luật, gán vòng theo ghế."""
    s = reset(cat, rng.make_seeds(seed, B), P)
    g = np.random.default_rng(seed)
    pols = []
    if bots:
        lineup = [bots[i % len(bots)] for i in range(P)]
        for n in dict.fromkeys(lineup):
            pols.append((ScriptedBot(cat, PRESETS[n]), np.array([i for i, x in enumerate(lineup) if x == n])))
    t_mask = t_bot = t_step = 0.0
    games = combats = 0
    for _ in range(steps):
        t0 = time.perf_counter()
        m = legal_actions(cat, s)
        t1 = time.perf_counter()
        if pols:
            a = np.full((B, P), END_TURN, np.int64)
            for pol, seats in pols:
                a[:, seats] = pol.act(s, m, seats)
        else:
            a = random_actions(m, g, end_bias=0.1)
        t2 = time.perf_counter()
        s, info = step(cat, combat, s, a, mask=m)
        t3 = time.perf_counter()
        t_mask += t1 - t0
        t_bot += t2 - t1
        t_step += t3 - t2
        games += int(info["reset"].sum())
        combats += int(info["combat"].sum())
    core = t_mask + t_step
    return dict(B=B, steps_per_s=B * steps / core, decisions_per_s=B * P * steps / core,
                mask_ms=1e3 * t_mask / steps, step_ms=1e3 * t_step / steps, bot_ms=1e3 * t_bot / steps,
                games=games, combat_frac=combats / (B * steps))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bots", default="", help="vd tempo,hyper,roll7,fast9 (mặc định: bot ngẫu nhiên)")
    ap.add_argument("--profile", action="store_true", help="in 15 hàm tốn thời gian nhất với B = 1024")
    a = ap.parse_args(argv)
    cat = load_catalog()
    combat = FormulaCombat(catalog=cat)
    bots = [b for b in a.bots.split(",") if b]
    who = ", ".join(bots) if bots else "bot ngẫu nhiên"
    print(f"P = {a.players}, {a.steps} bước mỗi cấu hình, {who}  (tốc độ chỉ tính lõi: mask + step)")
    print(f"{'B':>6} {'bước-ván/giây':>15} {'quyết định/giây':>16} {'mask ms':>9} {'step ms':>9}"
          f" {'bot ms':>8} {'% ván giao tranh':>17} {'ván xong':>9}")
    for B in (1, 64, 1024):
        r = run(cat, combat, B, a.players, a.steps, bots=bots)
        print(f"{r['B']:>6} {r['steps_per_s']:>15,.0f} {r['decisions_per_s']:>16,.0f} {r['mask_ms']:>9.2f}"
              f" {r['step_ms']:>9.2f} {r['bot_ms']:>8.2f} {100 * r['combat_frac']:>16.1f}% {r['games']:>9}")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        run(cat, combat, 1024, a.players, a.steps, bots=bots)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
