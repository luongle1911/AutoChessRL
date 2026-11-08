"""Hàm dựng dữ liệu dùng chung cho kiểm thử."""
from battlesim import from_dict


def champ(name, hp=100.0, ad=0.0, atk_speed=0.0, armor=0.0, rng=1, crit=0.0, crit_mult=1.0):
    return {"name": name, "hp": hp, "ad": ad, "as": atk_speed, "armor": armor, "mr": 0,
            "range": rng, "crit_chance": crit, "crit_mult": crit_mult,
            "mana_max": 100, "mana_start": 0, "skill": "single_damage",
            "skill_power": 0, "skill_param": 0, "traits": []}


def tiny_config(*champs):
    return from_dict({
        "global": {"dt": 0.1, "max_steps": 300, "move_time": 0.5,
                   "star_mult": {"hp": [1, 1.8, 3.24], "ad": [1, 1.5, 2.25]}},
        "traits": [],
        "champions": list(champs),
    })


def random_team(rng, n_champs, size):
    """Đội ngẫu nhiên: (mã tướng, sao, ô nửa sân) với ô không trùng."""
    cells = rng.choice(28, size=size, replace=False)
    return [(int(rng.integers(n_champs)), int(rng.integers(1, 4)), int(c)) for c in cells]


def bounce_counts(events, n_units=20):
    """Số lần liên tiếp mỗi tướng đi về đúng ô nó vừa rời (A -> B -> A), reset khi nó ra đòn."""
    from battlesim.layout import EV_ATTACK, EV_MOVE
    worst = [0] * n_units
    cur = [0] * n_units
    trail = [[] for _ in range(n_units)]
    for e in events:
        u, typ = int(e[2]), int(e[1])
        if typ == EV_ATTACK:
            cur[u] = 0
        elif typ == EV_MOVE:
            cell = int(e[5])
            if len(trail[u]) >= 2 and trail[u][-2] == cell:
                cur[u] += 1
                worst[u] = max(worst[u], cur[u])
            trail[u].append(cell)
    return worst
