"""Adapter: ba hàm thuần nối lõi với lớp trình bày.

to_view(catalog, state, b, seat, god=False, ...) -> dict  (view model, chỉ chứa thứ ghế đó được thấy)
diff_events(view_before, view_after) -> list[str]           (suy sự kiện từ hai view)
parse_command(text, view) -> int | None                      (lệnh gõ -> số hiệu hành động)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from autochess.core.actions import BUY0, SELL0, MOVE0, ROLL, BUY_XP, END_TURN, NUM_ACTIONS
from autochess.core.catalog import BENCH_SIZE, NUM_SLOTS, SHOP_SIZE, MAX_LEVEL
from autochess.core.catalog import NO_BREAKPOINT, STAT_NAMES
from autochess.core.combat import board_features, unit_final_stats
from autochess.core.pairing import meet_prob

STARS = {1: "★", 2: "★★", 3: "★★★"}


def champ_stats(catalog, c: int, star: int) -> dict | None:
    """Chỉ số gốc của tướng c ở sao `star` (1..3), chưa tính tộc hệ. None với luật cũ (không có chỉ số)."""
    if catalog.stats is None:
        return None
    v = catalog.stats[c].astype(float).copy()
    v[:3] *= float(catalog.star_mult[star - 1])
    return {k: int(round(x)) for k, x in zip(STAT_NAMES, v)}


def _unit(catalog, code: int, slot: int, final=None) -> dict | None:
    if code < 0:
        return None
    c, st = code // 3, code % 3 + 1
    d = dict(slot=slot, champ=c, name=catalog.names[c], star=st, cost=int(catalog.cost[c]),
             power=float(catalog.power[c, st - 1]),
             traits=[catalog.trait_names[k] for k in np.flatnonzero(catalog.trait_matrix[c])])
    if catalog.stats is not None:
        d["stats"] = champ_stats(catalog, c, st)
        if final is not None:     # quân trên sân: chỉ số sau hiệu ứng tộc hệ
            d["final_stats"] = {k: int(round(x)) for k, x in zip(STAT_NAMES, final)}
    return d


def _traits(catalog, feats, p: int) -> list:
    out = []
    for k in range(catalog.K):
        n = int(feats["trait_count"][0, p, k])
        if n == 0:
            continue
        bp = [int(x) for x in catalog.trait_breakpoints[k].tolist() if x < NO_BREAKPOINT]
        lvl = int(feats["trait_level"][0, p, k])
        nxt = bp[lvl] if lvl < len(bp) else bp[-1]
        out.append(dict(id=k, name=catalog.trait_names[k], count=n, level=lvl, breakpoints=bp, next=nxt))
    out.sort(key=lambda t: (-t["level"], -t["count"], t["id"]))
    return out


def _player(catalog, sb, feats, p: int, full: bool) -> dict:
    lv = int(sb.level[0, p])
    units = sb.units[0, p]
    final = unit_final_stats(catalog, units[BENCH_SIZE:][None], None)[0] if catalog.stats is not None else None
    d = dict(
        seat=p, hp=int(sb.hp[0, p]), gold=int(sb.gold[0, p]), level=lv, xp=int(sb.xp[0, p]),
        xp_next=int(catalog.xp_to_next[lv - 1]) if lv < MAX_LEVEL else 0,
        streak=int(sb.streak[0, p]), alive=bool(sb.alive[0, p]), placement=int(sb.placement[0, p]),
        ended=bool(sb.ended_turn[0, p]),
        actions_left=int(catalog.max_actions_per_round - sb.actions_this_round[0, p]),
        board=[_unit(catalog, int(units[j]), j, None if final is None else final[j - BENCH_SIZE])
               for j in range(BENCH_SIZE, NUM_SLOTS)],
        board_count=int(feats["n_board"][0, p]),
        strength=float(feats["strength"][0, p]),
        traits=_traits(catalog, feats, p),
        last_result=dict(result=int(sb.last_result[0, p, 0]), hp_lost=int(sb.last_result[0, p, 1]),
                         opponent=int(sb.last_opponent[0, p])),
        hidden=not full,
        team=None if "team_hp" not in feats else dict(
            hp=int(feats["team_hp"][0, p]), ad=int(feats["team_ad"][0, p]), ap=int(feats["team_ap"][0, p]),
            armor=int(round(feats["team_armor"][0, p])), mr=int(round(feats["team_mr"][0, p]))),
        # hàng chờ là thông tin công khai như ĐTCL (agent từ giai đoạn 3 cũng thấy); shop vẫn ẩn
        bench=[_unit(catalog, int(units[j]), j) for j in range(BENCH_SIZE)],
    )
    if full:
        owned = Counter(int(u) // 3 for u in units if u >= 0)
        d["shop"] = [None if c < 0 else dict(slot=i, champ=int(c), name=catalog.names[c],
                                              cost=int(catalog.cost[c]), owned=owned.get(int(c), 0),
                                              stats=champ_stats(catalog, int(c), 1),
                                              traits=[catalog.trait_names[k] for k in
                                                      np.flatnonzero(catalog.trait_matrix[c])])
                     for i, c in enumerate(sb.shop[0, p].tolist())]
        d["interest_next"] = int(min(d["gold"] // catalog.interest_step, catalog.interest_max))
    return d


def to_view(catalog, state, b: int, seat: int, god: bool = False, combat=None,
            matchup_row: bool = True) -> dict:
    """View model của ván b nhìn từ ghế `seat`. god=True: thấy toàn bộ."""
    sb = state.rows(b)
    P = sb.P
    feats = board_features(catalog, sb)
    mp = meet_prob(sb)[0]
    win = combat.evaluate(catalog, sb)[0][0] if combat is not None else None
    me = _player(catalog, sb, feats, seat, full=True)
    view = dict(
        round=int(sb.round[0]), num_players=P, seat=seat, god=god, done=bool(sb.done[0]),
        me=me, shop=me.pop("shop"), bench=me.pop("bench"), board=me["board"],
        matchup_row=matchup_row and win is not None,
        opponents=[],
    )
    exp = 0.0
    for k in range(1, P):
        j = (seat + k) % P
        o = _player(catalog, sb, feats, j, full=god)
        o["meet_prob"] = float(mp[seat, j])
        if view["matchup_row"]:
            o["win_prob"] = float(win[seat, j])
            exp += o["meet_prob"] * o["win_prob"]
        view["opponents"].append(o)
    view["expected_win"] = exp if view["matchup_row"] and me["alive"] and not view["done"] else None
    if god:
        total = catalog.pool_size(P)
        view["pool"] = [dict(champ=c, name=catalog.names[c], cost=int(catalog.cost[c]),
                             left=int(sb.pool[0, c]), total=int(total[c])) for c in range(catalog.N)]
        view["matchup"] = win.tolist() if win is not None else None
        view["meet_prob"] = mp.tolist()
        view["seed"] = int(sb.seed[0])
    return view


# ---------------------------------------------------------------- diff
def _unit_bag(view) -> Counter:
    return Counter((u["name"], u["star"]) for u in view["board"] + view["bench"] if u)


def diff_events(before: dict, after: dict) -> list[str]:
    """Suy sự kiện của chính ghế đó từ hai view liên tiếp."""
    out = []
    a, b = before["me"], after["me"]
    if after["round"] != before["round"]:
        out.append(f"sang vòng {after['round']}")
    ub, ua = _unit_bag(before), _unit_bag(after)
    gained, lost = ua - ub, ub - ua
    for (name, st), n in sorted(gained.items()):
        out.append(f"+{n} {name} {STARS[st]}")
    for (name, st), n in sorted(lost.items()):
        out.append(f"-{n} {name} {STARS[st]}")
    for (name, st), n in gained.items():
        if st >= 2 and lost.get((name, st - 1), 0) >= 2:
            out.append(f"ghép: {name} {STARS[st - 1]} -> {name} {STARS[st]}")
    if b["level"] != a["level"]:
        out.append(f"lên cấp {b['level']}")
    if b["hp"] != a["hp"]:
        out.append(f"máu {a['hp']} -> {b['hp']} ({b['hp'] - a['hp']:+d})")
    if b["gold"] != a["gold"]:
        out.append(f"vàng {a['gold']} -> {b['gold']} ({b['gold'] - a['gold']:+d})")
    if a["alive"] and not b["alive"]:
        out.append(f"bị loại, hạng {b['placement']}")
    return out


# ---------------------------------------------------------------- lệnh
_ALIASES = {
    "b": "b", "buy": "b", "mua": "b",
    "s": "s", "sell": "s", "ban": "s", "bán": "s",
    "m": "m", "move": "m", "chuyen": "m", "chuyển": "m",
    "r": "r", "roll": "r", "d": "r",
    "x": "x", "xp": "x", "f": "x", "lv": "x",
    "e": "e", "end": "e", "xong": "e",
    "a": "a", "act": "a",
}


def parse_command(text: str, view: dict | None = None) -> int | None:
    """Dịch lệnh gõ thành đúng một số hiệu hành động; None nếu không phải lệnh hành động.

    b i: mua ô shop i (0-4)     s j: bán ô j (0-17)     m j: chuyển ô j lên/xuống sân
    r: roll     x: mua XP     e: kết thúc lượt     a n: gửi thẳng số hiệu n (gỡ lỗi)
    """
    parts = text.strip().lower().split()
    if not parts or parts[0] not in _ALIASES:
        return None
    cmd, args = _ALIASES[parts[0]], parts[1:]
    if cmd in ("r", "x", "e"):
        return None if args else {"r": ROLL, "x": BUY_XP, "e": END_TURN}[cmd]
    if len(args) != 1 or not args[0].lstrip("-").isdigit():
        return None
    n = int(args[0])
    if cmd == "b":
        return BUY0 + n if 0 <= n < SHOP_SIZE else None
    if cmd == "s":
        return SELL0 + n if 0 <= n < NUM_SLOTS else None
    if cmd == "m":
        return MOVE0 + n if 0 <= n < NUM_SLOTS else None
    if cmd == "a":
        return n if 0 <= n < NUM_ACTIONS else None
    return None


def action_to_command(a: int) -> str:
    a = int(a)
    if a < SELL0:
        return f"b {a - BUY0}"
    if a < MOVE0:
        return f"s {a - SELL0}"
    if a < ROLL:
        return f"m {a - MOVE0}"
    return {ROLL: "r", BUY_XP: "x", END_TURN: "e"}[a]
