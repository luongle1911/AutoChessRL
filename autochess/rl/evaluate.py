"""Bộ đánh giá: seed cố định, agent bốc hành động theo phân phối của nó.

    python3 -m autochess.rl.evaluate --ckpt runs/<tên>/checkpoints/latest.pt [--games 1000]
    python3 -m autochess.rl.evaluate --ckpt <mạng A> --vs <mạng B>          # đấu trực tiếp hai mạng
    python3 -m autochess.rl.evaluate --ckpt <mạng> --catalog <file dữ liệu khác>   # thử "bản cập nhật"

Đo:
  - tỷ lệ thắng 1 đấu 1 trước từng đối thủ (đã đổi ghế), và theo giai đoạn (vòng 1-6, 7-11, 12-16, 17+);
  - Elo (Bradley-Terry) của agent trong cùng bảng với các bot (bảng bot-đấu-bot được tính một lần và lưu lại);
  - thống kê hành vi: roll/vòng, vàng khi kết thúc lượt, cấp theo vòng, số lần chạm trần hành động;
  - (tùy chọn) phân phối hạng trong ván nhiều người cùng bot.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import zlib

import numpy as np

from autochess import tournament as T

BOTS = ("tempo", "hyper", "roll7", "fast9")


def _seed(base, *names):
    return base + zlib.crc32("|".join(names).encode()) % 100_000


def bot_matrix(cat, combat, names=BOTS + ("random",), games: int = 400, seed: int = 777, B: int = 256,
               cache: str | None = None, players: dict | None = None) -> dict:
    """Số ván thắng giữa các đối thủ cố định (để neo bảng Elo). Lưu vào `cache` nếu có.
    players: {tên: người chơi} cho các đối thủ không phải bot luật (vd bản neo của agent)."""
    names = tuple(names)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        if d["names"] == list(names) and d["games"] == games and d.get("catalog") == cat.version:
            return d
    W = np.zeros((len(names), len(names)))
    for i, j in itertools.combinations(range(len(names)), 2):
        for lineup in ((names[i], names[j]), (names[j], names[i])):
            r = T.play(cat, combat, lineup, games // 2, seed=_seed(seed, *lineup), B=B, players=players)
            pl = r["placements"]
            a, b = names.index(lineup[0]), names.index(lineup[1])
            W[a, b] += (pl[:, 0] == 1).sum()
            W[b, a] += (pl[:, 1] == 1).sum()
    d = dict(names=list(names), games=games, catalog=cat.version, wins=W.tolist())
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w") as f:
            json.dump(d, f)
    return d


def load_anchors(cat, combat, anchors: dict, device="cpu", seed: int = 4321) -> dict:
    """{tên: checkpoint} -> {tên: AgentPlayer} (bản neo đóng băng dùng khi đánh giá)."""
    from .agent import AgentPlayer
    from .compat import load_compatible
    from .policy import Policy
    out = {}
    for k, (name, path) in enumerate((anchors or {}).items()):
        net, cfg, _ = load_compatible(path, cat, device)
        out[name] = AgentPlayer(Policy(net, device), cat, combat, matchup_row=cfg["env"]["matchup_row"],
                                seed=seed + k, name=name)
    return out


def evaluate(cat, combat, policy, opponents=BOTS + ("random",), games: int = 1000, seed: int = 12345,
             B: int = 256, matchup_row: bool = True, matrix: dict | None = None,
             ffa_players: tuple = (), ffa_games: int = 400, anchors: dict | None = None) -> dict:
    """anchors: {tên: AgentPlayer} các bản neo đóng băng (giai đoạn 2), đấu thêm sau các bot."""
    from .agent import AgentPlayer
    agent = AgentPlayer(policy, cat, combat, matchup_row=matchup_row, seed=seed)
    anchors = anchors or {}
    res = dict(winrate={}, phase={}, games={})
    wins = {}
    for opp in list(opponents) + [a for a in anchors if a not in opponents]:
        w = n = 0
        pw = np.zeros(len(T.PHASES))
        pn = np.zeros(len(T.PHASES))
        for lineup in (("agent", opp), (opp, "agent")):
            r = T.play(cat, combat, lineup, games // 2, seed=_seed(seed, *lineup), B=B,
                       players={"agent": agent, **anchors})
            seat = lineup.index("agent")
            pl = r["placements"]
            w += int((pl[:, seat] == 1).sum())
            n += len(pl)
            c = r["combats"]
            sel = c["i"] == seat
            for k, (_, lo, hi) in enumerate(T.PHASES):
                f = sel & (c["round"] >= lo) & (c["round"] <= hi)
                pw[k] += c["win"][f].sum()
                pn[k] += f.sum()
        res["winrate"][opp] = w / max(n, 1)
        res["games"][opp] = n
        res["phase"][opp] = [float(x) if m >= 20 else None for x, m in zip(pw / np.maximum(pn, 1), pn)]
        wins[opp] = (w, n)
    bots = [o for o in opponents if o != "random" and o not in anchors]
    res["mean_winrate_bots"] = float(np.mean([res["winrate"][o] for o in bots])) if bots else None
    res["min_winrate_bots"] = float(min(res["winrate"][o] for o in bots)) if bots else None
    res["behavior"] = agent.summary()

    if matrix is not None:
        names = list(matrix["names"]) + ["agent"]
        k = len(names)
        W = np.zeros((k, k))
        W[:-1, :-1] = np.asarray(matrix["wins"])
        for opp, (w, n) in wins.items():
            if opp in names:
                j = names.index(opp)
                W[-1, j] += w
                W[j, -1] += n - w
        elo = T.bradley_terry_elo(names, W)
        res["elo"] = {n: round(float(e), 1) for n, e in elo.items()}

    if ffa_players:
        lineup = ("agent",) + tuple(ffa_players)
        r = T.play(cat, combat, lineup, ffa_games, seed=_seed(seed, *lineup), B=B, players={"agent": agent})
        pl = r["placements"][:, 0]
        P = len(lineup)
        res["ffa"] = dict(players=P, mean_place=float(pl.mean()), first=float((pl == 1).mean()),
                          top_half=float((pl <= P // 2).mean()), last=float((pl == P).mean()))

    m = res["winrate"]
    res["milestones"] = dict(
        m1=bool("random" in m and m["random"] > 0.95),
        m2=bool(bots and min(m[o] for o in bots) > 0.5),
        m3=bool(bots and min(m[o] for o in bots) > 0.6))
    if anchors:
        res["anchors"] = {a: m[a] for a in anchors}
    return res


def evaluate_lobbies(cat, combat, policy, lobbies: dict, players: dict | None = None, games: int = 128,
                     seed: int = 2468, B: int = 64, matchup_row: bool = True, elo_anchor: str | None = None) -> dict:
    """Đánh giá trong các sảnh nhiều người cố định (giai đoạn 3). Agent ngồi ghế 0 (ghép cặp ngẫu nhiên
    đều nên ghế không quan trọng).

    lobbies: {tên sảnh: [tên đối thủ ở các ghế còn lại]}, tên là bot luật hoặc khóa trong `players`
    (vd bản neo từ load_anchors). Trả về cho từng sảnh: hạng TB, tỷ lệ top nửa (top 4 ở 8 người), tỷ lệ nhất,
    tỷ lệ chót; cộng Elo (Bradley-Terry) tính từ mọi cặp "ai xếp trên ai" trong mọi sảnh (mỗi ván P người
    cho P(P-1)/2 cặp, chia cho P-1 để mỗi ván có trọng số như một ván 1 đấu 1).
    """
    from .agent import AgentPlayer
    agent = AgentPlayer(policy, cat, combat, matchup_row=matchup_row, seed=seed)
    players = dict(players or {})
    names = ["agent"]
    for opps in lobbies.values():
        for n in opps:
            if n not in names:
                names.append(n)
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    res = dict(lobbies={})
    for lname, opps in lobbies.items():
        lineup = ("agent",) + tuple(opps)
        P = len(lineup)
        r = T.play(cat, combat, lineup, games, seed=_seed(seed, lname, *lineup), B=min(B, games),
                   players={"agent": agent, **players})
        pl = r["placements"]
        a = pl[:, 0]
        res["lobbies"][lname] = dict(players=P, games=int(len(pl)), mean_place=float(a.mean()),
                                     top_half=float((a <= P // 2).mean()), first=float((a == 1).mean()),
                                     last=float((a == P).mean()),
                                     stderr_place=float(a.std() / np.sqrt(max(len(a), 1))))
        for i, x in enumerate(lineup):
            for j, y in enumerate(lineup):
                if x != y:
                    W[idx[x], idx[y]] += (pl[:, i] < pl[:, j]).sum() / (P - 1)
    elo = T.bradley_terry_elo(names, W, anchor=elo_anchor or "random", anchor_elo=1500.0)
    res["elo"] = {n: round(float(e), 1) for n, e in elo.items()}
    res["behavior"] = agent.summary()
    return res


def format_lobbies(res: dict) -> str:
    lines = []
    for n, d in res["lobbies"].items():
        lines.append(f"sảnh {n} ({d['players']} người, {d['games']} ván): hạng TB {d['mean_place']:.2f} "
                     f"± {d['stderr_place']:.2f}, top {d['players'] // 2} {100 * d['top_half']:.0f}%, "
                     f"nhất {100 * d['first']:.0f}%, chót {100 * d['last']:.0f}%")
    lines.append("Elo sảnh: " + ", ".join(f"{n} {e:.0f}" for n, e in sorted(res["elo"].items(), key=lambda x: -x[1])))
    b = res["behavior"]
    lines.append(f"hành vi (sảnh): roll/vòng {b['rolls_per_round']:.2f}, XP/vòng {b['xp_per_round']:.2f}, "
                 f"mua/vòng {b['buys_per_round']:.2f}, vàng khi kết thúc lượt {b['gold_at_end_turn']:.1f}, "
                 f"cấp theo vòng {b['level_at_round']}")
    if "late_3star" in b:
        lines.append(late_line(b))
    return "\n".join(lines)


def late_line(b: dict) -> str:
    mt = ", ".join(f"{k} {100 * v:.0f}%" for k, v in b["main_traits"].items())
    return (f"cuối ván (vòng 20+): quân 3 sao {b['late_3star']:.2f}, 2 sao {b['late_2star']:.2f}, tộc hệ kích hoạt "
            f"{b['late_active_traits']:.2f}; lối chính: {mt} (độ đa dạng {b['comp_diversity']:.1f})")


def head_to_head(cat, combat, policy_a, policy_b, games: int = 1000, seed: int = 4242, B: int = 256,
                 matchup_row: bool = True, matchup_row_b: bool | None = None) -> dict:
    """Hai mạng đấu 1-1 (đổi ghế). Hai mạng có thể đọc hai bản quan sát khác nhau."""
    from .agent import AgentPlayer
    mb = matchup_row if matchup_row_b is None else matchup_row_b
    pa = AgentPlayer(policy_a, cat, combat, matchup_row=matchup_row, seed=seed, name="A")
    pb = AgentPlayer(policy_b, cat, combat, matchup_row=mb, seed=seed + 1, name="B")
    w = n = 0
    pw = np.zeros(len(T.PHASES))
    pn = np.zeros(len(T.PHASES))
    for lineup in (("A", "B"), ("B", "A")):
        r = T.play(cat, combat, lineup, games // 2, seed=_seed(seed, *lineup), B=B, players={"A": pa, "B": pb})
        seat = lineup.index("A")
        pl = r["placements"]
        w += int((pl[:, seat] == 1).sum())
        n += len(pl)
        c = r["combats"]
        sel = c["i"] == seat
        for k, (_, lo, hi) in enumerate(T.PHASES):
            f = sel & (c["round"] >= lo) & (c["round"] <= hi)
            pw[k] += c["win"][f].sum()
            pn[k] += f.sum()
    wr = w / max(n, 1)
    return dict(winrate_a=wr, games=n, stderr=float(np.sqrt(wr * (1 - wr) / max(n, 1))),
                phase_a=[float(x) if m >= 20 else None for x, m in zip(pw / np.maximum(pn, 1), pn)])


def format_eval(res: dict) -> str:
    anc = res.get("anchors", {})
    lines = ["tỷ lệ thắng 1 đấu 1: " + ", ".join(f"{o} {100 * w:.0f}%" for o, w in res["winrate"].items()
                                                 if o not in anc)]
    if anc:
        lines.append("trước bản neo: " + ", ".join(f"{o} {100 * w:.0f}%" for o, w in anc.items()))
    ph = []
    for o, v in res["phase"].items():
        ph.append(f"{o} [" + " ".join("-" if x is None else f"{100 * x:.0f}" for x in v) + "]")
    lines.append("theo giai đoạn (1-6, 7-11, 12-16, 17+): " + "; ".join(ph))
    if "elo" in res:
        lines.append("Elo: " + ", ".join(f"{n} {e:.0f}" for n, e in sorted(res["elo"].items(), key=lambda x: -x[1])))
    b = res["behavior"]
    lines.append(f"hành vi: roll/vòng {b['rolls_per_round']:.2f}, XP/vòng {b['xp_per_round']:.2f}, "
                 f"mua/vòng {b.get('buys_per_round', 0):.2f}, bán/vòng {b.get('sells_per_round', 0):.2f}, "
                 f"chuyển/vòng {b.get('moves_per_round', 0):.2f}, "
                 f"vàng khi kết thúc lượt {b['gold_at_end_turn']:.1f}, quyết định/vòng {b['decisions_per_round']:.1f}, "
                 f"chạm trần/vòng {b['cap_hits_per_round']:.3f}, cấp theo vòng {b['level_at_round']}")
    if "ffa" in res:
        f = res["ffa"]
        lines.append(f"{f['players']} người: hạng TB {f['mean_place']:.2f}, nhất {100 * f['first']:.0f}%, "
                     f"top nửa {100 * f['top_half']:.0f}%, chót {100 * f['last']:.0f}%")
    ms = res["milestones"]
    lines.append("mốc: " + ", ".join(f"{k} {'ĐẠT' if v else 'chưa'}" for k, v in ms.items()))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ffa", default="", help="vd tempo,hyper,roll7,fast9,tempo,hyper,roll7 cho ván 8 người")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vs", default=None, help="checkpoint thứ hai: đấu trực tiếp thay vì đấu bot")
    ap.add_argument("--catalog", default=None, help="file dữ liệu khác (mặc định catalog_v1.json)")
    ap.add_argument("--lobbies", default=None,
                    help="file YAML/JSON cấu hình (vd configs/p3_lobby.yaml): đánh giá trong các sảnh eval.lobbies "
                         "với bản neo eval.anchors của nó")
    a = ap.parse_args(argv)
    from autochess.core import load_catalog, FormulaCombat
    from .policy import Policy, load_checkpoint
    cat = load_catalog(a.catalog)
    combat = FormulaCombat(catalog=cat)
    net, cfg = load_checkpoint(a.ckpt, cat, a.device)
    if a.lobbies:
        from .config import load_config
        ec = load_config([a.lobbies])["eval"]
        anchors = load_anchors(cat, combat, ec.get("anchors") or {}, a.device)
        r = evaluate_lobbies(cat, combat, Policy(net, a.device), ec["lobbies"], players=anchors,
                             games=a.games, B=a.B, matchup_row=cfg["env"]["matchup_row"],
                             elo_anchor=ec.get("lobby_elo_anchor"))
        print(format_lobbies(r))
        return
    if a.vs:
        net_b, cfg_b = load_checkpoint(a.vs, cat, a.device)
        r = head_to_head(cat, combat, Policy(net, a.device), Policy(net_b, a.device), games=a.games, seed=a.seed,
                         B=a.B, matchup_row=cfg["env"]["matchup_row"], matchup_row_b=cfg_b["env"]["matchup_row"])
        ph = " ".join("-" if x is None else f"{100 * x:.0f}" for x in r["phase_a"])
        print(f"{a.ckpt}\n  thắng {100 * r['winrate_a']:.1f}% ± {100 * r['stderr']:.1f} trước {a.vs} "
              f"({r['games']} ván); theo giai đoạn (1-6, 7-11, 12-16, 17+): {ph}")
        return
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.ckpt))), "bot_matrix.json")
    matrix = bot_matrix(cat, combat, cache=cache)
    res = evaluate(cat, combat, Policy(net, a.device), games=a.games, seed=a.seed, B=a.B,
                   matchup_row=cfg["env"]["matchup_row"], matrix=matrix,
                   ffa_players=tuple(x for x in a.ffa.split(",") if x))
    print(format_eval(res))


if __name__ == "__main__":
    main()
