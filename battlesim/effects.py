"""Khối hiệu ứng dùng chung cho kỹ năng, tộc hệ, đồ và lõi (luật bản 5).

Một khối = KHI NÀO (trigger) + LÊN AI (target) + LÀM GÌ (kind), kèm số liệu theo bậc (sao của tướng / mốc
của tộc), thời hạn, cộng dồn. Trong file luật mỗi khối là một dict dễ đọc, ví dụ:

    {"when": "attack", "to": "self", "do": "stat", "stat": "as", "mode": "mul", "values": [0.05],
     "max_stacks": 99}                       # Cuồng Cung: +5% tốc đánh mỗi đòn, cộng dồn

compile_effects() đổi danh sách khối thành hai bảng số (EI nguyên, EF thực) cho trình mô phỏng.
Trình mô phỏng chỉ có một bộ thông dịch cho các bảng này; thêm đồ / lõi mới là thêm dữ liệu.

Số liệu (values) được nhân với nguồn tỷ lệ (scale): không (x1), công của người dùng, sức mạnh phép của người
dùng (/100), máu tối đa của mục tiêu, máu tối đa của người dùng, máu đã mất của mục tiêu.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- chỉ số (cộng / nhân được)
STATS = ("hp", "ad", "ap", "as", "armor", "mr", "crit_chance", "crit_dmg", "mana_max", "range",
         "vamp", "amp", "dr", "amp_big", "amp_shield", "ability_crit", "mana_per_attack", "mana_start")
(ST_HP, ST_AD, ST_AP, ST_AS, ST_ARMOR, ST_MR, ST_CRIT_CH, ST_CRIT_DMG, ST_MANA_MAX, ST_RANGE,
 ST_VAMP, ST_AMP, ST_DR, ST_AMP_BIG, ST_AMP_SHIELD, ST_ABIL_CRIT, ST_MANA_ATK, ST_MANA_START) = range(18)
NST = len(STATS)
STAT_VN = {"hp": "máu", "ad": "công", "ap": "sức mạnh phép", "as": "tốc đánh", "armor": "giáp",
           "mr": "kháng phép", "crit_chance": "tỷ lệ chí mạng", "crit_dmg": "sát thương chí mạng",
           "mana_max": "năng lượng tối đa", "range": "tầm đánh", "vamp": "hút máu", "amp": "sát thương gây ra",
           "dr": "giảm sát thương nhận", "amp_big": "sát thương lên mục tiêu nhiều máu",
           "amp_shield": "sát thương lên mục tiêu có khiên", "ability_crit": "kỹ năng chí mạng được",
           "mana_per_attack": "năng lượng mỗi đòn", "mana_start": "năng lượng khởi đầu"}

# ---------------------------------------------------------------- khi nào
TRIGGERS = ("start", "attack", "attack_n", "crit", "cast", "periodic", "at_time", "hp_below", "hit_taken",
            "kill")
(TR_START, TR_ATTACK, TR_ATTACK_N, TR_CRIT, TR_CAST, TR_PERIODIC, TR_AT_TIME, TR_HP_BELOW, TR_HIT_TAKEN,
 TR_KILL) = range(10)
# "period" (EF_PERIOD) nghĩa theo trigger: attack_n = mỗi N đòn; periodic = mỗi N giây; at_time = một lần ở
# giây N; hp_below = ngưỡng (0..1, một lần); hit_taken / attack / crit = thời gian hồi nội tại (0 = không)

# ---------------------------------------------------------------- lên ai
TARGETS = ("self", "target", "enemies_r", "allies_r", "around_target", "lowest_ally", "farthest_enemy",
           "nearest_n", "near_target_n", "densest", "all_allies", "row_allies", "all_enemies")
(TG_SELF, TG_TARGET, TG_ENEMIES_R, TG_ALLIES_R, TG_AROUND_TARGET, TG_LOWEST_ALLY, TG_FARTHEST,
 TG_NEAREST_N, TG_NEAR_TARGET_N, TG_DENSEST, TG_ALL_ALLIES, TG_ROW_ALLIES, TG_ALL_ENEMIES) = range(13)
# enemies_r / allies_r: trong bán kính r quanh mình (allies_r gồm cả mình); around_target: mục tiêu và kẻ địch
# trong bán kính r quanh nó; nearest_n: n kẻ địch gần mình nhất trong bán kính r; near_target_n: n kẻ địch gần
# mục tiêu nhất (trừ mục tiêu); densest: kẻ địch có nhiều kẻ địch nhất trong bán kính r quanh nó, cùng chúng

# ---------------------------------------------------------------- làm gì
KINDS = ("damage", "heal", "shield", "stun", "stat", "mana", "wound", "burn", "dash", "stealth", "cc_immune")
(K_DAMAGE, K_HEAL, K_SHIELD, K_STUN, K_STAT, K_MANA, K_WOUND, K_BURN, K_DASH, K_STEALTH, K_CC_IMMUNE) = range(11)
DTYPES = ("physical", "magic", "true")
MODES = ("add", "mul")
SCALES = ("none", "ad", "ap", "target_hp", "self_hp", "target_missing_hp")
(SC_NONE, SC_AD, SC_AP, SC_TGT_HP, SC_SELF_HP, SC_TGT_MISSING) = range(6)
CONDS = ("none", "back_rows", "front_rows")

# cột của EI (nguyên) và EF (thực)
(EI_TRIG, EI_TARGET, EI_KIND, EI_STAT, EI_MODE, EI_SCALE, EI_COUNT, EI_RADIUS, EI_STACKS, EI_COND,
 EI_KEY) = range(11)
NEI = 11
EF_V0, EF_DUR, EF_PERIOD = 0, 4, 5      # EF_V0..EF_V0+3: số liệu theo bậc 0..3
NEF = 6
MAX_LEVELS = 4


def _idx(table, name, what):
    if name not in table:
        raise ValueError(f"{what} không hợp lệ: {name!r} (có: {', '.join(table)})")
    return table.index(name)


def compile_effects(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Danh sách khối (dict) -> (EI [n, NEI] int32, EF [n, NEF] float32)."""
    n = len(rows)
    EI = np.zeros((n, NEI), np.int32)
    EF = np.zeros((n, NEF), np.float32)
    for i, r in enumerate(rows):
        EI[i, EI_TRIG] = _idx(TRIGGERS, r.get("when", "start"), "when")
        EI[i, EI_TARGET] = _idx(TARGETS, r.get("to", "self"), "to")
        kind = _idx(KINDS, r["do"], "do")
        EI[i, EI_KIND] = kind
        if kind == K_STAT:
            EI[i, EI_STAT] = _idx(STATS, r["stat"], "stat")
            EI[i, EI_MODE] = _idx(MODES, r.get("mode", "add"), "mode")
        elif kind == K_DAMAGE:
            EI[i, EI_MODE] = _idx(DTYPES, r.get("dtype", "magic"), "dtype")
        EI[i, EI_SCALE] = _idx(SCALES, r.get("scale", "none"), "scale")
        EI[i, EI_COUNT] = int(r.get("count", 1))
        EI[i, EI_RADIUS] = int(r.get("radius", 99))
        EI[i, EI_STACKS] = int(r.get("max_stacks", 1))
        EI[i, EI_COND] = _idx(CONDS, r.get("cond", "none"), "cond")
        EI[i, EI_KEY] = int(r.get("key", i))
        vals = list(r.get("values", [0.0]))
        if not 1 <= len(vals) <= MAX_LEVELS:
            raise ValueError(f"values phải có 1-{MAX_LEVELS} số: {r}")
        vals = vals + [vals[-1]] * (MAX_LEVELS - len(vals))
        EF[i, EF_V0:EF_V0 + MAX_LEVELS] = vals
        EF[i, EF_DUR] = float(r.get("duration", 0.0))
        EF[i, EF_PERIOD] = float(r.get("period", 0.0))
    return EI, EF


def is_static(r: dict) -> bool:
    """Khối chỉ cộng/nhân chỉ số cho chính mình suốt trận: gộp thẳng vào chỉ số gốc lúc dựng trận."""
    return (r.get("when", "start") == "start" and r.get("to", "self") == "self" and r["do"] == "stat"
            and not r.get("duration") and r.get("cond", "none") == "none" and r.get("max_stacks", 1) == 1)


def describe(r: dict, level: int | None = None) -> str:
    """Mô tả ngắn một khối bằng tiếng Việt (cho giao diện)."""
    vals = r.get("values", [0])
    v = vals if level is None else [vals[min(level, len(vals) - 1)]]
    vs = "/".join(_fmt(x, r) for x in v)
    when = {"start": "đầu trận", "attack": "mỗi đòn", "attack_n": f"mỗi {int(r.get('period', 1))} đòn",
            "crit": "khi chí mạng", "cast": "khi tung chiêu", "periodic": f"mỗi {r.get('period', 1):g} giây",
            "at_time": f"ở giây {r.get('period', 0):g}", "hp_below": f"khi xuống dưới {100 * r.get('period', 0):.0f}% máu",
            "hit_taken": "khi bị đánh", "kill": "khi hạ địch"}[r.get("when", "start")]
    to = {"self": "bản thân", "target": "mục tiêu", "enemies_r": f"kẻ địch trong {r.get('radius', 1)} ô",
          "allies_r": f"đồng minh trong {r.get('radius', 1)} ô", "around_target": "mục tiêu và kẻ địch quanh nó",
          "lowest_ally": "đồng minh ít máu nhất", "farthest_enemy": "kẻ địch xa nhất",
          "nearest_n": f"{r.get('count', 1)} kẻ địch gần nhất", "near_target_n": f"{r.get('count', 1)} kẻ địch cạnh mục tiêu",
          "densest": "vùng đông địch nhất", "all_allies": "cả đội", "row_allies": "đồng minh cùng hàng",
          "all_enemies": "mọi kẻ địch"}[r.get("to", "self")]
    do = r["do"]
    dur = f" trong {r['duration']:g} giây" if r.get("duration") else ""
    if do == "stat":
        txt = f"{'+' if not vs.startswith('-') else ''}{vs} {STAT_VN[r['stat']]}"
        if r.get("max_stacks", 1) > 1:
            txt += f" (cộng dồn tối đa {r['max_stacks']})"
    elif do == "damage":
        txt = f"{vs} sát thương {dict(physical='vật lý', magic='phép', true='chuẩn')[r.get('dtype', 'magic')]}"
    else:
        txt = {"heal": f"hồi {vs} máu", "shield": f"khiên {vs}", "stun": f"choáng {vs} giây",
               "mana": f"+{vs} năng lượng", "wound": "vết thương (giảm 33% hồi máu)", "burn": f"thiêu {vs} mỗi giây",
               "dash": "lướt tới", "stealth": "mất dấu", "cc_immune": "miễn khống chế"}[do]
        if do in ("stun",):
            dur = ""
    cond = {"none": "", "back_rows": " (nếu đứng 2 hàng sau)", "front_rows": " (nếu đứng 2 hàng trước)"}[r.get("cond", "none")]
    return f"{when}, {to}: {txt}{dur}{cond}"


def _fmt(x, r):
    sc = r.get("scale", "none")
    if sc == "ad":
        return f"{100 * x:.0f}% công"
    if sc == "ap":
        return f"{x:g}"
    if sc in ("target_hp", "self_hp", "target_missing_hp"):
        return f"{100 * x:g}% máu" + {"target_hp": " tối đa mục tiêu", "self_hp": " tối đa",
                                       "target_missing_hp": " đã mất"}[sc]
    if r.get("do") == "stat" and r.get("mode") == "mul":
        return f"{100 * x:g}%"
    if r.get("do") == "stat" and r.get("stat") in ("crit_chance", "crit_dmg", "vamp", "amp", "dr", "amp_big",
                                                   "amp_shield"):
        return f"{100 * x:g}%"
    return f"{x:g}"
