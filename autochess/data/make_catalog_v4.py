"""Sinh catalog_v4.json (luật bản 4): 59 tướng thần thoại, 5 chỉ số, 24 tộc hệ có hiệu ứng chỉ số.

    python3 -m autochess.data.make_catalog_v4          # ghi autochess/data/catalog_v4.json

File JSON là nguồn sự thật; script này chỉ để sinh lại khi cân bằng (sửa bảng dưới rồi chạy lại).

Chỉ số (1 sao): máu, sát thương vật lý (mỗi giây), sát thương phép (mỗi giây), giáp, kháng phép.
  - Mẫu theo vai trò (hệ) ở giá 1, nhân theo giá (COST_MULT) cho máu và sát thương; giáp và kháng phép cộng
    thêm theo giá. Mỗi tướng có thể lệch nhẹ khỏi mẫu (TWEAK).
  - Mỗi sao: máu và sát thương x1,8 (STAR_MULT), giáp và kháng phép giữ nguyên. Nhờ vậy 3 sao tướng 1 tiền
    ~ 2 sao tướng 4 tiền, 3 sao tướng 2 tiền ~ giữa 2 sao tướng 4 và 5 tiền.
Giao tranh "ttk-v1" (core/combat.py): đội nào hạ đội kia nhanh hơn thì nhiều khả năng thắng hơn.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = ("hp", "ad", "ap", "armor", "mr")

# ----------------------------------------------------------------------------- tộc hệ
# Hiệu ứng: [chỉ số, kiểu, giá trị theo từng mốc]; kiểu "pct" = nhân (1 + x/100) với máu / sát thương,
# "flat" = cộng thẳng (giáp, kháng phép). scope "members" = chỉ tướng mang tộc, "team" = cả đội trên sân.
CLASSES = [
    dict(name="Hộ vệ", kind="class", breakpoints=[2, 4, 6], scope="members",
         effects=[["armor", "flat", [25, 50, 90]], ["mr", "flat", [25, 50, 90]]],
         text="Tướng Hộ vệ được cộng giáp và kháng phép."),
    dict(name="Đấu sĩ", kind="class", breakpoints=[2, 4, 6, 8], scope="members",
         effects=[["hp", "pct", [20, 40, 65, 100]]],
         text="Tướng Đấu sĩ được tăng máu."),
    dict(name="Sát thủ", kind="class", breakpoints=[2, 4, 6], scope="members",
         effects=[["ad", "pct", [25, 50, 85]]],
         text="Tướng Sát thủ được tăng sát thương vật lý. Khắc Xạ thủ và Pháp sư (lao vào hàng sau), "
              "bị Hộ vệ khắc."),
    dict(name="Xạ thủ", kind="class", breakpoints=[2, 4, 6], scope="members",
         effects=[["ad", "pct", [20, 45, 75]]],
         text="Tướng Xạ thủ được tăng sát thương vật lý. Khắc Hộ vệ."),
    dict(name="Pháp sư", kind="class", breakpoints=[3, 5, 7], scope="members",
         effects=[["ap", "pct", [30, 60, 100]]],
         text="Tướng Pháp sư được tăng sát thương phép. Khắc Đấu sĩ (đánh lan vào đám đông)."),
]
ORIGINS = [
    dict(name="Hy Lạp", breakpoints=[3, 5, 7], scope="team", effects=[["ap", "pct", [10, 25, 45]]],
         text="Cả đội được tăng sát thương phép."),
    dict(name="Bắc Âu", breakpoints=[2, 4, 6], scope="members", effects=[["ad", "pct", [15, 35, 60]]],
         text="Tướng Bắc Âu được tăng sát thương vật lý."),
    dict(name="Ai Cập", breakpoints=[2, 4, 6], scope="team",
         effects=[["armor", "flat", [10, 25, 45]], ["mr", "flat", [10, 25, 45]]],
         text="Cả đội được cộng giáp và kháng phép."),
    dict(name="Trung Hoa", breakpoints=[3, 6], scope="members",
         effects=[["ad", "pct", [25, 60]], ["ap", "pct", [25, 60]]],
         text="Tướng Trung Hoa được tăng cả sát thương vật lý lẫn phép."),
    dict(name="Việt Nam", breakpoints=[2, 4, 6], scope="team", effects=[["hp", "pct", [8, 18, 32]]],
         text="Cả đội được tăng máu."),
    dict(name="Nhật Bản", breakpoints=[2, 4], scope="members",
         effects=[["armor", "flat", [30, 70]], ["ad", "pct", [10, 25]]],
         text="Tướng Nhật Bản được cộng giáp và tăng sát thương vật lý."),
    dict(name="Ấn Độ", breakpoints=[2, 3, 4, 5], scope="members", effects=[["ap", "pct", [10, 20, 35, 55]],
                                                                          ["ad", "pct", [10, 20, 35, 55]]],
         text="Tướng Ấn Độ được tăng sát thương; mỗi tướng thêm một mốc."),
    dict(name="Lưỡng Hà", breakpoints=[2, 3], scope="members", effects=[["hp", "pct", [25, 50]]],
         text="Tướng Lưỡng Hà được tăng máu."),
    dict(name="Celt", breakpoints=[2, 4], scope="team", effects=[["mr", "flat", [20, 45]]],
         text="Cả đội được cộng kháng phép."),
    dict(name="Trung Mỹ", breakpoints=[2, 4], scope="members",
         effects=[["ad", "pct", [20, 50]], ["hp", "pct", [10, 25]]],
         text="Tướng Trung Mỹ được tăng sát thương vật lý và máu."),
    dict(name="Quái thú", breakpoints=[3, 6], scope="members",
         effects=[["hp", "pct", [25, 60]], ["armor", "flat", [15, 35]]],
         text="Quái thú được tăng máu và giáp."),
]
UNIQUES = [  # tộc chỉ một tướng mang (tướng 5 tiền)
    dict(name="Vua Olympus", owner="Zeus", scope="team", effects=[["ap", "pct", [15]]],
         text="Zeus: cả đội tăng 15% sát thương phép."),
    dict(name="Toàn Phụ", owner="Odin", scope="team", effects=[["ap", "pct", [10]], ["mr", "flat", [10]]],
         text="Odin: cả đội tăng 10% sát thương phép, +10 kháng phép."),
    dict(name="Thái Dương", owner="Ra", scope="team", effects=[["ad", "pct", [10]]],
         text="Ra: cả đội tăng 10% sát thương vật lý."),
    dict(name="Vá Trời", owner="Nữ Oa", scope="team", effects=[["armor", "flat", [15]], ["mr", "flat", [15]]],
         text="Nữ Oa: cả đội +15 giáp và kháng phép."),
    dict(name="Rồng Tiên", owner="Lạc Long Quân", scope="team", effects=[["hp", "pct", [12]]],
         text="Lạc Long Quân: cả đội tăng 12% máu."),
    dict(name="Mẹ Hỗn Mang", owner="Tiamat", scope="members", effects=[["hp", "pct", [40]]],
         text="Tiamat: tự tăng 40% máu."),
    dict(name="Gương Khói", owner="Tezcatlipoca", scope="members", effects=[["ad", "pct", [40]]],
         text="Tezcatlipoca: tự tăng 40% sát thương vật lý."),
    dict(name="Đại Thần Mặt Trời", owner="Amaterasu", scope="team",
         effects=[["ad", "pct", [8]], ["ap", "pct", [8]]],
         text="Amaterasu: cả đội tăng 8% sát thương vật lý và phép."),
]

# Ma trận khắc chế giữa các hệ (hàng khắc cột, dương = hàng có lợi), phản đối xứng; đại diện cho phần
# đặt quân (sát thủ lao vào hàng sau...) mà công thức chỉ số không có. Agent xếp quân sau này có thể học thay.
ROLE_COUNTER = {
    ("Sát thủ", "Xạ thủ"): 0.30, ("Sát thủ", "Pháp sư"): 0.30, ("Hộ vệ", "Sát thủ"): 0.30,
    ("Đấu sĩ", "Sát thủ"): 0.15, ("Xạ thủ", "Hộ vệ"): 0.15, ("Pháp sư", "Đấu sĩ"): 0.15,
}

# ----------------------------------------------------------------------------- tướng
# (tên, giá, hệ, [tộc gốc...])
CHAMPS = [
    # 1 tiền
    ("Sobek", 1, "Hộ vệ", ["Ai Cập", "Quái thú"]),
    ("Chử Đồng Tử", 1, "Hộ vệ", ["Việt Nam"]),
    ("Xolotl", 1, "Hộ vệ", ["Trung Mỹ"]),
    ("Enkidu", 1, "Đấu sĩ", ["Lưỡng Hà"]),
    ("Na Tra", 1, "Đấu sĩ", ["Trung Hoa"]),
    ("Momotarō", 1, "Đấu sĩ", ["Nhật Bản"]),
    ("Hermes", 1, "Sát thủ", ["Hy Lạp"]),
    ("Tengu", 1, "Sát thủ", ["Nhật Bản"]),
    ("Valkyrie", 1, "Xạ thủ", ["Bắc Âu"]),
    ("Karna", 1, "Xạ thủ", ["Ấn Độ"]),
    ("Brigid", 1, "Pháp sư", ["Celt"]),
    ("Kitsune", 1, "Pháp sư", ["Nhật Bản", "Quái thú"]),
    ("Bastet", 1, "Pháp sư", ["Ai Cập"]),
    # 2 tiền
    ("Heimdall", 2, "Hộ vệ", ["Bắc Âu"]),
    ("Anubis", 2, "Hộ vệ", ["Ai Cập"]),
    ("Dagda", 2, "Hộ vệ", ["Celt"]),
    ("Heracles", 2, "Đấu sĩ", ["Hy Lạp"]),
    ("Thạch Sanh", 2, "Đấu sĩ", ["Việt Nam"]),
    ("Cú Chulainn", 2, "Đấu sĩ", ["Celt"]),
    ("Camazotz", 2, "Sát thủ", ["Trung Mỹ", "Quái thú"]),
    ("Garuda", 2, "Sát thủ", ["Ấn Độ", "Quái thú"]),
    ("Artemis", 2, "Xạ thủ", ["Hy Lạp"]),
    ("Arjuna", 2, "Xạ thủ", ["Ấn Độ"]),
    ("Freyja", 2, "Pháp sư", ["Bắc Âu"]),
    ("Thoth", 2, "Pháp sư", ["Ai Cập"]),
    ("Hằng Nga", 2, "Pháp sư", ["Trung Hoa"]),
    # 3 tiền
    ("Cộng Công", 3, "Hộ vệ", ["Trung Hoa"]),
    ("Jörmungandr", 3, "Hộ vệ", ["Bắc Âu", "Quái thú"]),
    ("Atlas", 3, "Hộ vệ", ["Hy Lạp"]),
    ("Ares", 3, "Đấu sĩ", ["Hy Lạp"]),
    ("Thánh Gióng", 3, "Đấu sĩ", ["Việt Nam"]),
    ("Gilgamesh", 3, "Đấu sĩ", ["Lưỡng Hà"]),
    ("Hanuman", 3, "Đấu sĩ", ["Ấn Độ"]),
    ("Fenrir", 3, "Sát thủ", ["Bắc Âu", "Quái thú"]),
    ("Morrígan", 3, "Sát thủ", ["Celt"]),
    ("Hậu Nghệ", 3, "Xạ thủ", ["Trung Hoa"]),
    ("Horus", 3, "Xạ thủ", ["Ai Cập"]),
    ("Medusa", 3, "Pháp sư", ["Hy Lạp", "Quái thú"]),
    ("Âu Cơ", 3, "Pháp sư", ["Việt Nam"]),
    # 4 tiền
    ("Athena", 4, "Hộ vệ", ["Hy Lạp"]),
    ("Sơn Tinh", 4, "Hộ vệ", ["Việt Nam"]),
    ("Thor", 4, "Đấu sĩ", ["Bắc Âu"]),
    ("Sekhmet", 4, "Đấu sĩ", ["Ai Cập"]),
    ("Susanoo", 4, "Đấu sĩ", ["Nhật Bản"]),
    ("Tôn Ngộ Không", 4, "Sát thủ", ["Trung Hoa"]),
    ("Ishtar", 4, "Sát thủ", ["Lưỡng Hà"]),
    ("Lugh", 4, "Xạ thủ", ["Celt"]),
    ("Huitzilopochtli", 4, "Xạ thủ", ["Trung Mỹ"]),
    ("Hades", 4, "Pháp sư", ["Hy Lạp"]),
    ("Thủy Tinh", 4, "Pháp sư", ["Việt Nam"]),
    ("Ravana", 4, "Pháp sư", ["Ấn Độ"]),
    # 5 tiền (mỗi tướng thêm một tộc riêng, xem UNIQUES)
    ("Zeus", 5, "Pháp sư", ["Hy Lạp"]),
    ("Odin", 5, "Pháp sư", ["Bắc Âu"]),
    ("Ra", 5, "Xạ thủ", ["Ai Cập"]),
    ("Nữ Oa", 5, "Hộ vệ", ["Trung Hoa"]),
    ("Lạc Long Quân", 5, "Đấu sĩ", ["Việt Nam"]),
    ("Tiamat", 5, "Đấu sĩ", ["Quái thú"]),
    ("Tezcatlipoca", 5, "Sát thủ", ["Trung Mỹ"]),
    ("Amaterasu", 5, "Xạ thủ", ["Nhật Bản"]),
]

# Mẫu chỉ số 1 sao ở giá 1 theo hệ: máu, sát thương vật lý/giây, sát thương phép/giây, giáp, kháng phép
ROLE_BASE = {
    "Hộ vệ": (650, 25, 15, 45, 45),
    "Đấu sĩ": (600, 45, 10, 35, 30),
    "Sát thủ": (450, 65, 10, 25, 25),
    "Xạ thủ": (450, 60, 5, 20, 20),
    "Pháp sư": (450, 15, 60, 20, 30),
}
COST_MULT = {1: 1.0, 2: 1.25, 3: 1.55, 4: 1.9, 5: 2.35}   # máu và sát thương
DEF_PER_COST = 5                                          # giáp / kháng phép thêm mỗi bậc giá
STAR_MULT = [1.0, 1.8, 3.24]
# lệch riêng từng tướng (nhân máu, nhân sát thương)
TWEAK = {
    "Heracles": (1.10, 0.95), "Atlas": (1.12, 0.85), "Thánh Gióng": (1.05, 1.0), "Thor": (1.0, 1.08),
    "Tôn Ngộ Không": (1.08, 0.97), "Hậu Nghệ": (0.95, 1.08), "Medusa": (1.05, 0.95), "Sơn Tinh": (1.1, 0.95),
    "Thủy Tinh": (0.95, 1.08), "Tiamat": (1.1, 0.95), "Zeus": (0.95, 1.08), "Fenrir": (1.05, 0.98),
}
POOL = {"2": [12, 10, 8, 5, 4], "8": [30, 25, 18, 10, 9]}   # như ĐTCL (Set 12 trở đi) cho 8 người


def build() -> dict:
    traits = []
    for t in CLASSES + ORIGINS:
        traits.append(dict(id=len(traits), name=t["name"], kind=t.get("kind", "origin"),
                           breakpoints=t["breakpoints"], scope=t["scope"], effects=t["effects"], text=t["text"]))
    for t in UNIQUES:
        traits.append(dict(id=len(traits), name=t["name"], kind="unique", breakpoints=[1], scope=t["scope"],
                           effects=t["effects"], text=t["text"]))
    tid = {t["name"]: t["id"] for t in traits}
    unique_of = {t["owner"]: t["name"] for t in UNIQUES}
    champs = []
    for i, (name, cost, role, origins) in enumerate(CHAMPS):
        hp, ad, ap, ar, mr = ROLE_BASE[role]
        m = COST_MULT[cost]
        th, td = TWEAK.get(name, (1.0, 1.0))
        stats = dict(hp=round(hp * m * th / 5) * 5, ad=round(ad * m * td), ap=round(ap * m * td),
                     armor=ar + DEF_PER_COST * (cost - 1), mr=mr + DEF_PER_COST * (cost - 1))
        tr = [tid[role]] + [tid[o] for o in origins] + ([tid[unique_of[name]]] if name in unique_of else [])
        champs.append(dict(id=i, name=name, cost=cost, role=role, traits=tr, stats=stats))
    roles = [t["name"] for t in CLASSES]
    C = [[0.0] * len(roles) for _ in roles]
    for (a, b), v in ROLE_COUNTER.items():
        C[roles.index(a)][roles.index(b)] = v
        C[roles.index(b)][roles.index(a)] = -v
    base = json.load(open(os.path.join(HERE, "catalog_v3.json"), encoding="utf-8"))
    out = dict(
        version="catalog-v4",
        note=("Luật bản 4 (07/11/2026): 59 tướng thần thoại (13/13/13/12/8 theo giá), mỗi tướng 5 chỉ số (máu, "
              "sát thương vật lý, sát thương phép, giáp, kháng phép); 24 tộc hệ: 5 hệ vai trò, 11 tộc gốc theo thần "
              "thoại, 8 tộc chỉ một tướng mang; hiệu ứng tộc cộng chỉ số. Giao tranh ttk-v1: đội hạ đối thủ nhanh "
              "hơn thì nhiều khả năng thắng hơn. Kho chung như ĐTCL. Kinh tế, máu mất, giới hạn hành động như "
              "catalog-v3.2. Sinh bằng autochess/data/make_catalog_v4.py."),
        stats_note="chỉ số 1 sao; mỗi sao máu và sát thương x star_mult, giáp và kháng phép giữ nguyên",
        stat_names=list(STATS), star_mult=STAR_MULT,
        champions=champs, traits=traits,
        role_counter_note="khắc chế giữa các hệ: hàng khắc cột; cộng vào logit theo tỷ lệ hệ của hai đội",
        role_counter=dict(roles=[tid[r] for r in roles], matrix=C),
        pool_size_by_cost_note="số bản mỗi tướng theo giá 1..5, theo số người chơi (nội suy ở giữa)",
        pool_size_by_cost=POOL,
        shop_odds_note=base["shop_odds_note"], shop_odds=base["shop_odds"], xp_to_next=base["xp_to_next"],
        economy=base["economy"], rules=base["rules"],
        stage_damage_note=base["stage_damage_note"], stage_damage=base["stage_damage"],
        unit_damage_scale_note=base["unit_damage_scale_note"], unit_damage_scale=base["unit_damage_scale"],
        combat={"ttk-v1": dict(temperature=0.34, armor_scale=100.0,
                               note="logit(i thắng j) = (log TTK(j hạ i) - log TTK(i hạ j) + khắc chế) / T")},
    )
    return out


def main():
    out = build()
    path = os.path.join(HERE, "catalog_v4.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"đã ghi {path}: {len(out['champions'])} tướng {dict(sorted(Counter(c['cost'] for c in out['champions']).items()))}, "
          f"{len(out['traits'])} tộc hệ")
    cnt = Counter(k for c in out["champions"] for k in c["traits"])
    for t in out["traits"]:
        n = cnt[t["id"]]
        assert n >= t["breakpoints"][-1], (t["name"], n, t["breakpoints"])
        print(f"  {t['name']:<18} {n:2d} tướng, mốc {t['breakpoints']}")


if __name__ == "__main__":
    main()
