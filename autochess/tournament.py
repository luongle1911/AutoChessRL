"""Giải đấu bot: kiểm tra cân bằng luật và tạo thước đo cho agent.

    python3 -m autochess.tournament                      # đủ bốn phần, ~vài phút
    python3 -m autochess.tournament --games 400 --quick  # nhanh hơn, ít ván hơn

In ra:
  1. Đường cong sức mạnh: sức mạnh đội trung bình (theo module giao tranh) của từng bot theo vòng.
  2. Tỷ lệ thắng TỪNG TRẬN theo giai đoạn (vòng 1-6, 7-11, 12-16, 17+), 1 đấu 1, đã đổi ghế.
  3. Tỷ lệ thắng CẢ VÁN 1 đấu 1, đã đổi ghế.
  4. Thứ hạng trung bình trong ván 4 và 8 người (mỗi bot 1 hoặc 2 ghế).
  5. Elo (Bradley-Terry) từ mọi kết quả cặp đôi; bot ngẫu nhiên được neo ở 1000.

Mọi số liệu đều tất định theo --seed.
"""
from __future__ import annotations

import argparse
import itertools
import time
import zlib

import numpy as np

from autochess.core import load_catalog, FormulaCombat, reset, step, legal_actions, rng, END_TURN
from autochess.core.combat import board_features
from autochess.bots.random_bot import random_actions
from autochess.bots.scripted import PRESETS, ScriptedBot

PHASES = (("vòng 1-6", 1, 6), ("vòng 7-11", 7, 11), ("vòng 12-16", 12, 16), ("vòng 17+", 17, 10_000))
ROUND_BINS = ((1, 4), (5, 6), (7, 8), (9, 11), (12, 14), (15, 17), (18, 21), (22, 50))


class RecordingCombat:
    """Bọc module giao tranh, ghi lại sức mạnh đội và vòng của mỗi lần giao tranh."""

    def __init__(self, inner):
        self.inner = inner
        self.version = inner.version
        self.last = None

    def evaluate(self, catalog, state):
        f = board_features(catalog, state)
        self.last = dict(strength=f["strength"].copy(), round=np.asarray(state.round).copy())
        return self.inner.evaluate(catalog, state)


class RandomBot:
    name = "random"

    def __init__(self, seed=0):
        self.g = np.random.default_rng(seed)

    def act(self, state, mask, seats):
        return random_actions(mask[:, seats], self.g, end_bias=0.1)


def make_policy(cat, name, seed=0):
    return RandomBot(seed) if name == "random" else ScriptedBot(cat, PRESETS[name])


def play(cat, combat, lineup, games, seed=0, B=256, max_steps=200_000, players=None):
    """Chạy các ván với đội hình ghế cố định `lineup` (tên bot theo ghế) tới khi đủ `games` ván.

    players: tùy chọn {tên: đối tượng có act(state, mask, seats) -> [B, len(seats)]}, ví dụ agent.

    Trả về dict: placements [G, P], combats (mảng có cấu trúc), steps, seconds.
    """
    P = len(lineup)
    rec = RecordingCombat(combat)
    players = players or {}
    pols = {n: players[n] if n in players else make_policy(cat, n, seed) for n in dict.fromkeys(lineup)}
    seats_of = {n: np.array([i for i, x in enumerate(lineup) if x == n]) for n in pols}
    s = reset(cat, rng.make_seeds(seed, B), P)
    places, cr, ci, cj, cw, cs_i, cs_j = [], [], [], [], [], [], []
    # Mỗi ô của lô đóng góp đúng k ván trọn vẹn: không thiên vị ván ngắn (ván ngắn xong trước).
    k = -(-games // B)
    slot_games = np.zeros(B, np.int64)
    steps = 0
    t0 = time.perf_counter()
    while (slot_games < k).any() and steps < max_steps:
        m = legal_actions(cat, s)
        a = np.full((s.B, P), END_TURN, np.int64)
        for n, pol in pols.items():
            a[:, seats_of[n]] = pol.act(s, m, seats_of[n])
        s, info = step(cat, rec, s, a, mask=m)
        steps += 1
        if info["combat"].any():
            bi = np.flatnonzero(info["combat"])
            st, rd = rec.last["strength"], rec.last["round"]
            opp, gh = info["opp"][bi], info["ghost"][bi]
            res = s.last_result[bi, :, 0]
            for i in range(P):
                j = opp[:, i].astype(np.int64)
                f = (j >= 0) & ~gh[:, i] & (slot_games[bi] < k)
                if not f.any():
                    continue
                cr.append(rd[f]); ci.append(np.full(f.sum(), i)); cj.append(j[f])
                cw.append(res[f, i] > 0); cs_i.append(st[f, i]); cs_j.append(st[f][np.arange(f.sum()), j[f]])
        rs = info["reset"]
        if rs.any():
            take = rs & (slot_games < k)
            places.append(info["final_placement"][take])
            slot_games[rs] += 1
    sec = time.perf_counter() - t0
    cat_ = lambda xs, dt=None: np.concatenate(xs) if xs else np.zeros(0, dt)
    combats = dict(round=cat_(cr, np.int64), i=cat_(ci, np.int64), j=cat_(cj, np.int64),
                   win=cat_(cw, bool), s_i=cat_(cs_i, np.float64), s_j=cat_(cs_j, np.float64))
    pl = np.concatenate(places) if places else np.zeros((0, P))
    return dict(lineup=list(lineup), placements=pl, combats=combats, steps=steps, seconds=sec,
                decisions=steps * B * P)


# ---------------------------------------------------------------- phân tích
def bradley_terry_elo(names, wins, anchor="random", anchor_elo=1000.0, iters=2000):
    """wins[a, b] = số lần a thắng b. Trả về Elo (thang 400·log10)."""
    n = len(names)
    w = wins + 0.5 * (np.ones((n, n)) - np.eye(n))   # tiên nghiệm nhẹ để tránh vô cực
    g = np.ones(n)
    tot = w + w.T
    for _ in range(iters):
        denom = (tot / (g[:, None] + g[None, :])).sum(1)
        g_new = w.sum(1) / denom
        g_new /= np.exp(np.log(g_new).mean())
        if np.allclose(g_new, g, rtol=1e-10):
            g = g_new
            break
        g = g_new
    elo = 400 * np.log10(g)
    if anchor in names:
        elo += anchor_elo - elo[names.index(anchor)]
    else:
        elo += 1500 - elo.mean()
    return dict(zip(names, elo))


def table(rows, header, fmt=None):
    widths = [max(len(str(h)), *(len(str(r[k])) for r in rows)) for k, h in enumerate(header)]
    line = "  ".join(f"{h:>{w}}" for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(f"{str(v):>{w}}" for v, w in zip(r, widths)))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", default="tempo,hyper,roll7,fast9")
    ap.add_argument("--games", type=int, default=1000, help="số ván cho mỗi cặp 1 đấu 1 (mỗi chiều ghế)")
    ap.add_argument("--ffa-games", type=int, default=1000, help="số ván 4 và 8 người")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", default=None, help="file dữ liệu khác (mặc định catalog_v1.json)")
    ap.add_argument("--B", type=int, default=256, help="số ván chạy song song")
    ap.add_argument("--quick", action="store_true", help="giảm số ván 4 lần")
    ap.add_argument("--no-random", action="store_true", help="bỏ bot ngẫu nhiên khỏi bảng Elo")
    a = ap.parse_args(argv)
    if a.quick:
        a.games, a.ffa_games = max(50, a.games // 4), max(50, a.ffa_games // 4)

    cat = load_catalog(a.catalog)
    combat = FormulaCombat(catalog=cat)
    print(f"Dữ liệu: {cat.version}")
    bots = a.bots.split(",")
    elo_names = bots + ([] if a.no_random else ["random"])
    idx = {n: k for k, n in enumerate(elo_names)}
    W = np.zeros((len(elo_names), len(elo_names)))
    t_all = time.perf_counter()

    # ------------------------------------------------ 1 đấu 1, đổi ghế
    curve = {n: {b: [] for b in ROUND_BINS} for n in bots}
    phase_w = {(x, y, ph[0]): [0, 0] for x in bots for y in bots for ph in PHASES}
    game_w = {(x, y): [0, 0] for x in bots for y in bots}
    speed = []
    pairs = list(itertools.combinations(elo_names, 2))
    for x, y in pairs:
        for lineup in ((x, y), (y, x)):
            r = play(cat, combat, lineup, a.games, seed=a.seed + zlib.crc32(','.join(lineup).encode()) % 10_000, B=a.B)
            speed.append((r["decisions"], r["seconds"], r["steps"] * a.B))
            pl = r["placements"]
            for seat, name in enumerate(lineup):
                other = lineup[1 - seat]
                won = int((pl[:, seat] == 1).sum())
                W[idx[name], idx[other]] += won
                if name in bots and other in bots:
                    game_w[(name, other)][0] += won
                    game_w[(name, other)][1] += len(pl)
            c = r["combats"]
            for seat, name in enumerate(lineup):
                if name not in bots:
                    continue
                sel = c["i"] == seat
                for lo, hi in ROUND_BINS:
                    f = sel & (c["round"] >= lo) & (c["round"] <= hi)
                    curve[name][(lo, hi)].append(c["s_i"][f])
                other = lineup[1 - seat]
                if other not in bots:
                    continue
                for ph, lo, hi in PHASES:
                    f = sel & (c["round"] >= lo) & (c["round"] <= hi)
                    phase_w[(name, other, ph)][0] += int(c["win"][f].sum())
                    phase_w[(name, other, ph)][1] += int(f.sum())

    print("\n=== 1. Sức mạnh đội trung bình theo vòng (1 đấu 1, chỉ tính trận đã diễn ra) ===")
    rows = []
    for lo, hi in ROUND_BINS:
        vals = []
        for n in bots:
            v = np.concatenate(curve[n][(lo, hi)]) if curve[n][(lo, hi)] else np.zeros(0)
            vals.append(v.mean() if v.size >= 20 else np.nan)
        best = bots[int(np.nanargmax(vals))] if not np.all(np.isnan(vals)) else "-"
        rows.append([f"{lo}-{hi}"] + [("-" if np.isnan(v) else f"{v:.0f}") for v in vals] + [best])
    print(table(rows, ["vòng"] + bots + ["mạnh nhất"]))

    print("\n=== 2. Tỷ lệ thắng từng trận theo giai đoạn (hàng thắng cột, %) ===")
    for ph, _, _ in PHASES:
        rows, avg = [], []
        for x in bots:
            cells, tot_w, tot_n = [], 0, 0
            for y in bots:
                w, n = phase_w[(x, y, ph)]
                if x == y:
                    cells.append("·")
                    continue
                cells.append(f"{100 * w / n:.0f}" if n >= 20 else "-")
                tot_w, tot_n = tot_w + w, tot_n + n
            avg.append(tot_w / tot_n if tot_n else np.nan)
            rows.append([x] + cells + [f"{100 * avg[-1]:.0f}" if tot_n else "-"])
        best = bots[int(np.nanargmax(avg))] if not np.all(np.isnan(avg)) else "-"
        print(f"\n[{ph}]  mạnh nhất: {best}")
        print(table(rows, ["bot"] + bots + ["TB"]))

    print("\n=== 3. Tỷ lệ thắng cả ván 1 đấu 1 (hàng thắng cột, %, đã đổi ghế) ===")
    rows = []
    for x in bots:
        cells = []
        for y in bots:
            w, n = game_w[(x, y)]
            cells.append("·" if x == y else f"{100 * w / n:.0f}")
        rows.append([x] + cells)
    print(table(rows, ["bot"] + bots))

    # ------------------------------------------------ nhiều người
    print("\n=== 4. Thứ hạng trung bình nhiều người (thấp hơn = tốt hơn) ===")
    ffa_rows = []
    for P in (4, 8):
        reps = P // len(bots) if P % len(bots) == 0 else None
        lineup = (bots * (P // len(bots) + 1))[:P] if reps is None else bots * reps
        r = play(cat, combat, lineup, a.ffa_games, seed=a.seed + P, B=a.B)
        speed.append((r["decisions"], r["seconds"], r["steps"] * a.B))
        pl = r["placements"]
        for i, x in enumerate(lineup):
            for j, y in enumerate(lineup):
                if x != y:
                    W[idx[x], idx[y]] += (pl[:, i] < pl[:, j]).sum() / (P - 1)
        avg = {n: np.mean([pl[:, i].mean() for i, x in enumerate(lineup) if x == n]) for n in bots}
        top = {n: np.mean([(pl[:, i] <= P // 2).mean() for i, x in enumerate(lineup) if x == n]) for n in bots}
        for n in bots:
            ffa_rows.append([f"{P} người", n, f"{avg[n]:.2f}", f"{100 * top[n]:.0f}"])
    print(table(ffa_rows, ["sảnh", "bot", "hạng TB", "top nửa %"]))

    print("\n=== 5. Elo (Bradley-Terry trên mọi kết quả cặp đôi) ===")
    elo = bradley_terry_elo(elo_names, W)
    rows = [[n, f"{e:.0f}"] for n, e in sorted(elo.items(), key=lambda kv: -kv[1])]
    print(table(rows, ["bot", "Elo"]))

    dec = sum(d for d, _, _ in speed)
    sec = sum(s for _, s, _ in speed)
    gs = sum(g for _, _, g in speed)
    print(f"\nTốc độ khi chạy bot luật: {gs / sec:,.0f} bước-ván/giây, {dec / sec:,.0f} quyết định/giây "
          f"(B = {a.B}, gồm cả thời gian bot). Tổng thời gian: {time.perf_counter() - t_all:.0f} giây.")


if __name__ == "__main__":
    main()
