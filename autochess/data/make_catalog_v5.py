"""Sinh catalog_v5.json, phần giao tranh của luật bản 5: 59 tướng (chỉ số kiểu ĐTCL + kỹ năng), 24 tộc hệ, 8
mảnh đồ + 36 đồ ghép, tất cả viết bằng khối hiệu ứng (battlesim/effects.py).

    python3 -m autochess.data.make_catalog_v5        # ghi autochess/data/catalog_v5.json

Tên tướng, tộc hệ, thành viên và mốc giữ như bản 4. Kinh tế / shop / kho chung chép từ catalog_v4_1 (phần môi
trường bản 5 — lõi, vòng quái, chọn tướng chung — thêm sau). Số liệu là bản khởi đầu, sẽ chỉnh bằng phép thử
cân bằng dùng trình mô phỏng.
"""
from __future__ import annotations

import json
import os

from . import make_catalog_v4 as V4
from .make_catalog_v4_1 import EFFECTS_V41

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- chỉ số tướng
# 1 sao, giá 1, theo hệ: máu, công, tốc đánh, giáp, kháng phép, tầm, năng lượng (khởi đầu, tối đa)
ROLE_STATS = {
    "Hộ vệ": dict(hp=650, ad=45, as_=0.60, armor=40, mr=40, range=1, mana=(40, 90)),
    "Đấu sĩ": dict(hp=650, ad=55, as_=0.65, armor=35, mr=30, range=1, mana=(30, 80)),
    "Sát thủ": dict(hp=500, ad=60, as_=0.75, armor=25, mr=25, range=1, mana=(20, 60)),
    "Xạ thủ": dict(hp=450, ad=55, as_=0.75, armor=20, mr=20, range=4, mana=(0, 60)),
    "Pháp sư": dict(hp=450, ad=35, as_=0.65, armor=20, mr=25, range=4, mana=(30, 70)),
}
HP_AD_BY_COST = {1: 1.0, 2: 1.15, 3: 1.35, 4: 1.6, 5: 1.9}
DEF_PER_COST = 3
SKILL_BY_COST = {1: 1.0, 2: 1.25, 3: 1.6, 4: 2.0, 5: 2.6}   # độ lớn kỹ năng theo giá
STAR_MULT = {"hp": [1.0, 1.8, 3.24], "ad": [1.0, 1.5, 2.25]}


def _star(x, cost, r=1):
    """Số liệu kỹ năng theo sao: x1 / x1,5 / x2,25 (tướng 5 tiền 3 sao x4)."""
    s3 = 4.0 if cost == 5 else 2.25
    return [round(x, r), round(x * 1.5, r), round(x * s3, r)]


def _dmg(base, cost):
    return _star(base * SKILL_BY_COST[cost], cost, 0)


# ----------------------------------------------------------------------------- mẫu kỹ năng
# mỗi mẫu: hàm (giá) -> danh sách khối hiệu ứng "khi tung chiêu"
def _cast(**kw):
    kw.setdefault("when", "cast")
    return kw


SKILLS = {
    # --- 1-2 tiền: một mục tiêu, một hiệu ứng
    "strike": lambda c: [_cast(to="target", do="damage", dtype="physical", scale="ad",
                               values=[2.0 + 0.1 * c, 2.1 + 0.1 * c, 2.2 + 0.1 * c])],
    "bolt": lambda c: [_cast(to="target", do="damage", dtype="magic", scale="ap", values=_dmg(220, c))],
    "shield_self": lambda c: [_cast(to="self", do="shield", scale="ap", values=_dmg(260, c), duration=4)],
    "heal_self": lambda c: [_cast(to="self", do="heal", scale="ap", values=_dmg(230, c))],
    "stun_short": lambda c: [_cast(to="target", do="stun", values=[1.5, 1.75, 2.0]),
                             _cast(to="target", do="damage", dtype="magic", scale="ap", values=_dmg(100, c))],
    "frenzy": lambda c: [_cast(to="self", do="stat", stat="as", mode="mul", values=[0.6, 0.7, 0.85],
                               duration=4)],
    # --- 3 tiền: hai thành phần hoặc vùng nhỏ
    "slam_stun": lambda c: [_cast(to="target", do="damage", dtype="magic", scale="ap", values=_dmg(200, c)),
                            _cast(to="target", do="stun", values=[1.5, 1.75, 2.0])],
    "cleave": lambda c: [_cast(to="around_target", radius=1, do="damage", dtype="physical", scale="ad",
                               values=[1.6, 1.7, 1.9])],
    "heal_ally": lambda c: [_cast(to="lowest_ally", do="heal", scale="ap", values=_dmg(280, c))],
    "ward": lambda c: [_cast(to="allies_r", radius=1, do="shield", scale="ap", values=_dmg(200, c), duration=4)],
    "ambush": lambda c: [_cast(to="farthest_enemy", do="dash"),
                         _cast(to="farthest_enemy", do="damage", dtype="physical", scale="ad",
                               values=[2.4, 2.5, 2.7])],
    "leap_stun": lambda c: [_cast(to="farthest_enemy", do="dash"),
                            _cast(to="farthest_enemy", do="stun", values=[1.5, 1.75, 2.0]),
                            _cast(to="farthest_enemy", do="damage", dtype="magic", scale="ap", values=_dmg(150, c))],
    "fan": lambda c: [_cast(to="nearest_n", count=3, radius=6, do="damage", dtype="physical", scale="ad",
                            values=[1.3, 1.4, 1.5])],
    "petrify": lambda c: [_cast(to="around_target", radius=1, do="stun", values=[1.5, 1.75, 2.0]),
                          _cast(to="around_target", radius=1, do="damage", dtype="magic", scale="ap",
                                values=_dmg(130, c))],
    # --- 4-5 tiền: diện rộng hoặc đa năng
    "nova": lambda c: [_cast(to="densest", radius=2, do="damage", dtype="magic", scale="ap", values=_dmg(220, c))],
    "dark_nova": lambda c: [_cast(to="densest", radius=2, do="damage", dtype="magic", scale="ap",
                                  values=_dmg(200, c)),
                            _cast(to="densest", radius=2, do="wound", values=[6], duration=6)],
    "flood": lambda c: [_cast(to="densest", radius=2, do="damage", dtype="magic", scale="ap", values=_dmg(190, c)),
                        _cast(to="densest", radius=2, do="stat", stat="as", mode="mul", values=[-0.3], duration=4)],
    "quake": lambda c: [_cast(to="enemies_r", radius=1, do="stun", values=[1.5, 2.0, 2.5]),
                        _cast(to="enemies_r", radius=1, do="damage", dtype="magic", scale="ap", values=_dmg(150, c))],
    "barrier": lambda c: [_cast(to="allies_r", radius=2, do="shield", scale="ap", values=_dmg(220, c), duration=5)],
    "barrier_all": lambda c: [_cast(to="all_allies", do="shield", scale="ap", values=_dmg(170, c), duration=6)],
    "thunder": lambda c: [_cast(to="around_target", radius=1, do="damage", dtype="magic", scale="ap",
                                values=_dmg(170, c)),
                          _cast(to="target", do="stun", values=[1.0, 1.25, 1.5])],
    "drain": lambda c: [_cast(to="enemies_r", radius=2, do="damage", dtype="magic", scale="ap", values=_dmg(120, c)),
                        _cast(to="self", do="heal", scale="ap", values=_dmg(200, c))],
    "cleave_big": lambda c: [_cast(to="around_target", radius=1, do="damage", dtype="physical", scale="ad",
                                   values=[2.2, 2.4, 2.8])],
    "clones": lambda c: [_cast(to="self", do="stealth", values=[1.5]),
                         _cast(to="nearest_n", count=3, radius=3, do="damage", dtype="physical", scale="ad",
                               values=[1.8, 2.0, 2.4])],
    "bull": lambda c: [_cast(to="farthest_enemy", do="dash"),
                       _cast(to="farthest_enemy", do="damage", dtype="physical", scale="ad", values=[2.8, 3.0, 3.4]),
                       _cast(to="farthest_enemy", do="stun", values=[1.0, 1.25, 1.5])],
    "volley": lambda c: [_cast(to="nearest_n", count=4, radius=8, do="damage", dtype="physical", scale="ad",
                               values=[1.4, 1.5, 1.7])],
    "fire_line": lambda c: [_cast(to="around_target", radius=1, do="damage", dtype="physical", scale="ad",
                                  values=[1.8, 1.9, 2.1]),
                            _cast(to="around_target", radius=1, do="burn", scale="target_hp", values=[0.02],
                                  duration=4)],
    "chain": lambda c: [_cast(to="nearest_n", count=4, radius=8, do="damage", dtype="magic", scale="ap",
                              values=_dmg(160, c))],
    "storm": lambda c: [_cast(to="densest", radius=1, do="damage", dtype="magic", scale="ap", values=_dmg(110, c))
                        for _ in range(3)],
    "rally": lambda c: [_cast(to="all_allies", do="stat", stat="as", mode="mul", values=[0.3, 0.4, 0.8],
                              duration=6),
                        _cast(to="all_allies", do="stat", stat="ap", mode="add", values=[20, 30, 60], duration=6)],
    "sun_boat": lambda c: [_cast(to="around_target", radius=2, do="damage", dtype="physical", scale="ad",
                                 values=[1.8, 2.0, 2.6]),
                           _cast(to="around_target", radius=2, do="burn", scale="target_hp", values=[0.02],
                                 duration=5)],
    "avatar": lambda c: [_cast(to="self", do="stat", stat="hp", mode="mul", values=[0.3, 0.4, 0.8], key=9001),
                         _cast(to="self", do="stat", stat="ad", mode="mul", values=[0.4, 0.5, 1.0], key=9002),
                         _cast(to="enemies_r", radius=1, do="damage", dtype="magic", scale="ap", values=_dmg(120, c))],
    "chaos_sea": lambda c: [_cast(to="enemies_r", radius=2, do="damage", dtype="magic", scale="ap", values=_dmg(150, c)),
                            _cast(to="enemies_r", radius=2, do="stat", stat="armor", mode="mul", values=[-0.3],
                                  duration=5),
                            _cast(to="self", do="heal", scale="ap", values=_dmg(250, c))],
    "smoke_mirror": lambda c: [_cast(to="self", do="stealth", values=[2.0]),
                               _cast(to="farthest_enemy", do="dash"),
                               _cast(to="around_target", radius=1, do="damage", dtype="physical", scale="ad",
                                     values=[2.5, 2.8, 4.0])],
    "sunlight": lambda c: [_cast(to="all_enemies", do="damage", dtype="magic", scale="ap", values=_dmg(110, c))],
}

# tên tướng -> (mẫu kỹ năng, tên kỹ năng, mô tả ngắn theo tích truyện)
CHAMP_SKILL = {
    "Sobek": ("shield_self", "Da Cá Sấu", "Thần cá sấu sông Nile khoác lớp da dày: tạo khiên cho bản thân."),
    "Chử Đồng Tử": ("heal_self", "Gậy Và Nón Thần", "Gậy và nón thần nhà sư cho: hồi máu cho bản thân."),
    "Xolotl": ("stun_short", "Dẫn Lối Âm Giới", "Chó thần dẫn hồn xuống Mictlan: làm choáng mục tiêu."),
    "Enkidu": ("strike", "Sức Hoang Dã", "Người rừng do thần nặn từ đất sét: đòn đánh mạnh."),
    "Na Tra": ("stun_short", "Càn Khôn Quyển", "Ném vòng càn khôn: làm choáng và gây sát thương phép."),
    "Momotarō": ("heal_self", "Bánh Kê", "Ăn bánh kê mẹ làm: hồi máu."),
    "Hermes": ("strike", "Dép Có Cánh", "Sứ giả nhanh nhất Olympus: đâm mục tiêu."),
    "Tengu": ("frenzy", "Quạt Lông", "Quạt lông thần gió: tăng mạnh tốc đánh."),
    "Valkyrie": ("strike", "Chọn Người Ngã Xuống", "Nữ thần chọn chiến binh tử trận: bắn mạnh."),
    "Karna": ("shield_self", "Giáp Mặt Trời", "Bộ giáp bẩm sinh của con thần Mặt Trời: tạo khiên."),
    "Brigid": ("bolt", "Lửa Thiêng", "Ngọn lửa thiêng của nữ thần Celt: cầu lửa vào mục tiêu."),
    "Kitsune": ("bolt", "Hồ Hỏa", "Lửa cáo chín đuôi: sát thương phép."),
    "Bastet": ("bolt", "Vuốt Mèo Thần", "Nữ thần mèo vồ: sát thương phép."),
    "Heimdall": ("stun_short", "Kèn Gjallarhorn", "Thổi kèn báo tận thế: làm choáng mục tiêu."),
    "Anubis": ("shield_self", "Cân Linh Hồn", "Cân trái tim với lông vũ Maat: tạo khiên."),
    "Dagda": ("heal_self", "Vạc Bất Tận", "Vạc không bao giờ cạn: hồi máu."),
    "Heracles": ("strike", "Mười Hai Kỳ Công", "Cú đấm của người hùng mạnh nhất Hy Lạp."),
    "Thạch Sanh": ("strike", "Rìu Thần", "Lưỡi rìu thần Thái tử Ngọc Hoàng dạy võ: chém mạnh."),
    "Cú Chulainn": ("strike", "Gáe Bolg", "Ngọn giáo gai không thể rút ra: đâm cực mạnh."),
    "Camazotz": ("strike", "Dơi Tử Thần", "Dơi thần cắt đầu: đòn chí mạng."),
    "Garuda": ("frenzy", "Cánh Bão", "Vua loài chim vỗ cánh: tăng tốc đánh."),
    "Artemis": ("strike", "Mũi Tên Bạc", "Mũi tên bạc của nữ thần săn bắn."),
    "Arjuna": ("frenzy", "Cung Gandiva", "Giương cung thần: bắn liên hồi."),
    "Freyja": ("bolt", "Áo Lông Chim Ưng", "Phép seiðr: sát thương phép."),
    "Thoth": ("bolt", "Chữ Thần", "Thần chữ viết khắc lời nguyền: sát thương phép."),
    "Hằng Nga": ("bolt", "Ánh Trăng", "Tia sáng cung Quảng Hàn: sát thương phép."),
    "Cộng Công": ("slam_stun", "Húc Núi Bất Chu", "Thần nước húc gãy cột trời: gây sát thương và choáng."),
    "Jörmungandr": ("ward", "Quấn Quanh Trái Đất", "Rắn quấn thế giới bao bọc đồng đội: khiên cho đồng minh kề."),
    "Atlas": ("ward", "Đỡ Bầu Trời", "Titan gánh trời che chở đồng đội: khiên cho đồng minh kề."),
    "Ares": ("cleave", "Cuồng Chiến", "Thần chiến tranh chém quanh mục tiêu."),
    "Thánh Gióng": ("slam_stun", "Nhổ Tre Đánh Giặc", "Quật bụi tre: sát thương và choáng."),
    "Gilgamesh": ("cleave", "Vua Uruk", "Chém quét quanh mục tiêu."),
    "Hanuman": ("leap_stun", "Nhảy Qua Biển", "Vượt biển sang Lanka: lướt tới kẻ địch xa nhất và làm choáng."),
    "Fenrir": ("ambush", "Nuốt Chửng", "Sói khổng lồ lao vào kẻ địch xa nhất."),
    "Morrígan": ("ambush", "Quạ Báo Tử", "Hóa quạ lao tới kẻ địch xa nhất."),
    "Hậu Nghệ": ("fan", "Bắn Rụng Mặt Trời", "Bắn ba mũi tên vào ba kẻ địch gần nhất."),
    "Horus": ("fan", "Mắt Horus", "Mắt chim ưng: bắn ba mục tiêu."),
    "Medusa": ("petrify", "Hóa Đá", "Ánh mắt hóa đá mục tiêu và kẻ đứng cạnh."),
    "Âu Cơ": ("heal_ally", "Mẹ Trăm Trứng", "Che chở con: hồi máu đồng minh ít máu nhất."),
    "Athena": ("quake", "Khiên Aegis", "Giơ khiên có đầu Medusa: làm choáng kẻ địch xung quanh."),
    "Sơn Tinh": ("barrier", "Dời Non", "Nâng núi chặn nước: khiên cho đồng minh trong 2 ô."),
    "Thor": ("thunder", "Búa Mjölnir", "Búa sấm giáng xuống mục tiêu, lan sang kẻ địch cạnh đó."),
    "Sekhmet": ("drain", "Khát Máu", "Nữ thần sư tử hút máu kẻ địch xung quanh."),
    "Susanoo": ("cleave_big", "Kiếm Kusanagi", "Thanh kiếm lấy từ đuôi rắn tám đầu: chém quanh mục tiêu."),
    "Tôn Ngộ Không": ("clones", "Phân Thân", "Nhổ lông hóa phân thân: mất dấu rồi đánh ba kẻ địch gần."),
    "Ishtar": ("bull", "Bò Trời", "Thả Bò Trời: lướt tới kẻ địch xa nhất, sát thương lớn và choáng."),
    "Lugh": ("volley", "Giáo Lửa Areadbhair", "Ngọn giáo lửa tự bay: đánh bốn kẻ địch gần nhất."),
    "Huitzilopochtli": ("fire_line", "Rắn Lửa Xiuhcoatl", "Rắn lửa thiêu mục tiêu và kẻ đứng cạnh."),
    "Hades": ("dark_nova", "Âm Phủ", "Mở cửa âm phủ vào vùng đông địch: sát thương và vết thương."),
    "Thủy Tinh": ("flood", "Nước Dâng", "Dâng nước vào vùng đông địch: sát thương và làm chậm."),
    "Ravana": ("chain", "Mười Đầu", "Mười cái đầu niệm chú: sát thương phép bốn kẻ địch."),
    "Zeus": ("storm", "Sấm Sét Olympus", "Gọi sét ba lần vào vùng đông địch nhất."),
    "Odin": ("rally", "Hiến Mắt Cầu Trí", "Ban trí tuệ: cả đội tăng tốc đánh và sức mạnh phép."),
    "Ra": ("sun_boat", "Thuyền Mặt Trời", "Thuyền mặt trời thiêu đốt vùng quanh mục tiêu."),
    "Nữ Oa": ("barrier_all", "Vá Trời", "Đá ngũ sắc vá trời: khiên cho cả đội."),
    "Lạc Long Quân": ("avatar", "Hóa Rồng", "Hiện nguyên hình rồng: tăng máu, công và đánh quanh mình."),
    "Tiamat": ("chaos_sea", "Biển Hỗn Mang", "Biển mặn nguyên thủy: sát thương, giảm giáp quanh mình, hồi máu."),
    "Tezcatlipoca": ("smoke_mirror", "Gương Khói", "Biến vào khói rồi xuất hiện sau lưng kẻ địch xa nhất."),
    "Amaterasu": ("sunlight", "Ánh Dương", "Bước ra khỏi hang: ánh sáng đốt mọi kẻ địch."),
}

# ----------------------------------------------------------------------------- tộc hệ
def _stat(stat, mode, vals, **kw):
    return dict(do="stat", stat=stat, mode=mode, values=vals, **kw)


ROLE_TRAITS = {
    "Hộ vệ": [_stat("armor", "add", [20, 40, 70]), _stat("mr", "add", [20, 40, 70]),
              dict(when="start", to="allies_r", radius=1, do="shield", values=[100, 200, 350], duration=8)],
    "Đấu sĩ": [_stat("hp", "mul", [0.12, 0.25, 0.40, 0.60]),
               dict(when="kill", to="self", do="heal", scale="self_hp", values=[0.15])],
    "Sát thủ": [dict(when="start", to="farthest_enemy", do="dash"),
                _stat("ability_crit", "add", [1]), _stat("crit_chance", "add", [0.10, 0.20, 0.35]),
                _stat("crit_dmg", "add", [0.10, 0.25, 0.40])],
    "Xạ thủ": [dict(when="attack", to="self", do="stat", stat="as", mode="mul", values=[0.04, 0.07, 0.11],
                    max_stacks=10)],
    "Pháp sư": [_stat("ap", "add", [20, 40, 70]), _stat("mana_start", "add", [10, 20, 30])],
}
_PCT = {"hp": ("hp", "mul", 0.01), "ad": ("ad", "mul", 0.01), "ap": ("ap", "add", 1.0),
        "armor": ("armor", "add", 1.0), "mr": ("mr", "add", 1.0)}


def _trait_effects(name, kind, v4_effects):
    if name in ROLE_TRAITS:
        return ROLE_TRAITS[name]
    out = []
    effs = EFFECTS_V41.get(name, v4_effects)
    for stat, mode, vals in effs:
        s, m, f = _PCT[stat]
        out.append(_stat(s, m, [round(v * f, 3) for v in vals]))
    if name == "Celt":
        out.append(dict(when="periodic", period=3, to="self", do="heal", scale="self_hp", values=[0.02, 0.04]))
    if name == "Ai Cập":
        out.append(dict(when="start", to="self", do="shield", values=[60, 150, 300], duration=6))
    return out


# ----------------------------------------------------------------------------- đồ
COMPONENTS = [
    ("Kiếm", [_stat("ad", "mul", [0.10])]),
    ("Cung", [_stat("as", "mul", [0.10])]),
    ("Gậy", [_stat("ap", "add", [10])]),
    ("Giáp", [_stat("armor", "add", [20])]),
    ("Áo choàng", [_stat("mr", "add", [20])]),
    ("Đai", [_stat("hp", "add", [150])]),
    ("Nước mắt", [_stat("mana_start", "add", [15])]),
    ("Găng", [_stat("crit_chance", "add", [0.20])]),
]
_C = {n: i for i, (n, _) in enumerate(COMPONENTS)}
# (mảnh a, mảnh b, tên, nhóm, khối hiệu ứng riêng)
COMBINED = [
    ("Kiếm", "Kiếm", "Kiếm Tử Thần", "Sát thương", [_stat("ad", "mul", [0.35])]),
    ("Kiếm", "Cung", "Diệt Khổng Lồ", "Sát thương theo máu", [_stat("amp_big", "add", [0.25])]),
    ("Kiếm", "Gậy", "Kiếm Súng", "Hút máu", [_stat("vamp", "add", [0.22])]),
    ("Kiếm", "Giáp", "Áo Choàng Bóng Đêm", "Phòng thủ",
     [dict(when="hp_below", period=0.6, to="self", do="stealth", values=[1.0]),
      dict(when="hp_below", period=0.6, to="self", do="stat", stat="as", mode="mul", values=[0.15])]),
    ("Kiếm", "Áo choàng", "Huyết Kiếm", "Hút máu",
     [_stat("vamp", "add", [0.20]),
      dict(when="hp_below", period=0.4, to="self", do="shield", scale="self_hp", values=[0.25], duration=5)]),
    ("Kiếm", "Đai", "Găng Bạo Lực", "Phòng thủ",
     [dict(when="hp_below", period=0.6, to="self", do="stat", stat="hp", mode="mul", values=[0.25]),
      dict(when="hp_below", period=0.6, to="self", do="stat", stat="ad", mode="mul", values=[0.35])]),
    ("Kiếm", "Nước mắt", "Thương Thần", "Năng lượng", [_stat("mana_per_attack", "add", [5])]),
    ("Kiếm", "Găng", "Vô Cực Kiếm", "Chí mạng", [_stat("ability_crit", "add", [1]), _stat("crit_dmg", "add", [0.10])]),
    ("Cung", "Cung", "Cuồng Cung", "Tốc đánh",
     [dict(when="attack", to="self", do="stat", stat="as", mode="mul", values=[0.05], max_stacks=99)]),
    ("Cung", "Gậy", "Nanh Rắn", "Tốc đánh",
     [dict(when="cast", to="self", do="stat", stat="as", mode="mul", values=[0.60], duration=5)]),
    ("Cung", "Giáp", "Ý Chí Titan", "Phòng thủ",
     [dict(when="attack", to="self", do="stat", stat="ad", mode="mul", values=[0.02], max_stacks=25, key=7001),
      dict(when="attack", to="self", do="stat", stat="ap", mode="add", values=[2], max_stacks=25, key=7001),
      dict(when="hit_taken", to="self", do="stat", stat="ad", mode="mul", values=[0.02], max_stacks=25, key=7001),
      dict(when="hit_taken", to="self", do="stat", stat="ap", mode="add", values=[2], max_stacks=25, key=7001)]),
    ("Cung", "Áo choàng", "Cung Bão", "Đánh lan",
     [dict(when="attack", to="near_target_n", count=1, do="damage", dtype="physical", scale="ad", values=[0.55])]),
    ("Cung", "Đai", "Tù Và Chiến", "Hỗ trợ",
     [dict(when="start", to="row_allies", do="stat", stat="as", mode="mul", values=[0.10], key=7002)]),
    ("Cung", "Nước mắt", "Dao Điện", "Xé kháng phép",
     [dict(when="attack_n", period=3, to="nearest_n", count=3, radius=4, do="damage", dtype="magic", values=[70]),
      dict(when="attack_n", period=3, to="nearest_n", count=3, radius=4, do="stat", stat="mr", mode="mul",
           values=[-0.30], duration=5, key=7003)]),
    ("Cung", "Găng", "Lời Thì Thầm", "Xé giáp",
     [dict(when="crit", to="target", do="stat", stat="armor", mode="mul", values=[-0.30], duration=3, key=7004)]),
    ("Gậy", "Gậy", "Mũ Phù Thủy", "Sát thương", [_stat("ap", "add", [50])]),
    ("Gậy", "Giáp", "Vương Miện Hộ Vệ", "Phòng thủ",
     [dict(when="start", to="self", do="shield", scale="self_hp", values=[0.25], duration=8),
      dict(when="at_time", period=8, to="self", do="stat", stat="ap", mode="add", values=[30])]),
    ("Gậy", "Áo choàng", "Ngọc Lôi", "Xé kháng phép",
     [dict(when="periodic", period=0.5, to="enemies_r", radius=2, do="stat", stat="mr", mode="mul", values=[-0.30],
           duration=0.6, key=7005)]),
    ("Gậy", "Đai", "Bùa Dịch Hạch", "Vết thương",
     [dict(when="cast", to="target", do="wound", values=[10], duration=10),
      dict(when="cast", to="target", do="burn", scale="target_hp", values=[0.01], duration=10)]),
    ("Gậy", "Nước mắt", "Trượng Thiên Thần", "Sức mạnh phép theo thời gian",
     [dict(when="periodic", period=5, to="self", do="stat", stat="ap", mode="add", values=[30], max_stacks=99)]),
    ("Gậy", "Găng", "Găng Ngọc", "Chí mạng", [_stat("ability_crit", "add", [1]), _stat("crit_dmg", "add", [0.10])]),
    ("Giáp", "Giáp", "Áo Gai", "Phòng thủ",
     [_stat("dr", "add", [0.08]),
      dict(when="hit_taken", period=2, to="enemies_r", radius=1, do="damage", dtype="magic", values=[80])]),
    ("Giáp", "Áo choàng", "Giáp Thạch Quỷ", "Phòng thủ",
     [dict(when="hit_taken", to="self", do="stat", stat="armor", mode="add", values=[2], max_stacks=25, key=7006),
      dict(when="hit_taken", to="self", do="stat", stat="mr", mode="add", values=[2], max_stacks=25, key=7006)]),
    ("Giáp", "Đai", "Áo Choàng Lửa", "Vết thương",
     [dict(when="periodic", period=2, to="nearest_n", count=1, radius=2, do="burn", scale="target_hp",
           values=[0.01], duration=3),
      dict(when="periodic", period=2, to="nearest_n", count=1, radius=2, do="wound", values=[3], duration=3)]),
    ("Giáp", "Nước mắt", "Tim Băng", "Khống chế",
     [dict(when="periodic", period=0.5, to="enemies_r", radius=2, do="stat", stat="as", mode="mul", values=[-0.20],
           duration=0.6, key=7007)]),
    ("Giáp", "Găng", "Giáp Kiên Định", "Phòng thủ", [_stat("dr", "add", [0.15])]),
    ("Áo choàng", "Áo choàng", "Vuốt Rồng", "Hồi máu",
     [dict(when="periodic", period=2, to="self", do="heal", scale="self_hp", values=[0.05])]),
    ("Áo choàng", "Đai", "Khăn Hư Vô", "Xé giáp và kháng",
     [dict(when="start", to="enemies_r", radius=2, do="stat", stat="armor", mode="mul", values=[-0.30], duration=10,
           key=7008),
      dict(when="start", to="enemies_r", radius=2, do="stat", stat="mr", mode="mul", values=[-0.30], duration=10,
           key=7008)]),
    ("Áo choàng", "Nước mắt", "Mũ Thích Nghi", "Năng lượng theo thời gian",
     [dict(when="periodic", period=3, to="self", do="mana", values=[10], cond="back_rows"),
      dict(when="start", to="self", do="stat", stat="armor", mode="add", values=[30], cond="front_rows"),
      dict(when="start", to="self", do="stat", stat="mr", mode="add", values=[30], cond="front_rows")]),
    ("Áo choàng", "Găng", "Khăn Giải Thuật", "Miễn khống chế",
     [dict(when="start", to="self", do="cc_immune", values=[14]),
      dict(when="periodic", period=1, to="self", do="stat", stat="as", mode="mul", values=[0.03], max_stacks=14)]),
    ("Đai", "Đai", "Giáp Máu", "Phòng thủ", [_stat("hp", "mul", [0.12])]),
    ("Đai", "Nước mắt", "Ngọc Cứu Rỗi", "Hỗ trợ, hồi máu",
     [dict(when="periodic", period=5, to="allies_r", radius=1, do="heal", scale="target_missing_hp", values=[0.10])]),
    ("Đai", "Găng", "Phá Khiên", "Sát thương", [_stat("amp_shield", "add", [0.25])]),
    ("Nước mắt", "Nước mắt", "Ngọc Lam", "Năng lượng",
     [_stat("mana_max", "add", [-10]), dict(when="cast", to="self", do="mana", values=[10])]),
    ("Nước mắt", "Găng", "Bàn Tay Công Lý", "Hút máu, sát thương",
     [_stat("ad", "mul", [0.15]), _stat("ap", "add", [15]), _stat("vamp", "add", [0.15])]),
    ("Găng", "Găng", "Găng Thợ Săn", "Chí mạng, năng lượng",
     [_stat("crit_chance", "add", [0.20]), dict(when="crit", to="self", do="mana", values=[5])]),
]


# ----------------------------------------------------------------------------- lõi
def _team(*blocks):
    return [dict(when=b.pop("when", "start"), to=b.pop("to", "self"), **b) for b in blocks]


AUGMENTS = [
    # kinh tế
    dict(name="Tiền Cúng", tier=1, group="Kinh tế", text="Nhận ngay 10 vàng.", econ=dict(gold_now=10)),
    dict(name="Lộc Đầu Năm", tier=2, group="Kinh tế", text="Nhận ngay 20 vàng.", econ=dict(gold_now=20)),
    dict(name="Kho Báu Rồng", tier=3, group="Kinh tế", text="Nhận ngay 35 vàng.", econ=dict(gold_now=35)),
    dict(name="Lương Bổng", tier=1, group="Kinh tế", text="+1 vàng mỗi vòng.", econ=dict(gold_per_round=1)),
    dict(name="Thương Nhân", tier=2, group="Kinh tế", text="+2 vàng mỗi vòng.", econ=dict(gold_per_round=2)),
    dict(name="Ngân Khố", tier=2, group="Kinh tế", text="Lãi tối đa +2 (7 vàng).", econ=dict(interest_max=2)),
    dict(name="Học Giả", tier=1, group="Kinh tế", text="+2 kinh nghiệm mỗi vòng.", econ=dict(xp_per_round=2)),
    dict(name="Thầy Bói", tier=2, group="Kinh tế", text="1 lần làm mới shop miễn phí mỗi vòng.",
         econ=dict(free_rolls=1)),
    dict(name="Thần Tài", tier=3, group="Kinh tế", text="Nhận ngay 10 vàng; 2 lần làm mới miễn phí mỗi vòng.",
         econ=dict(gold_now=10, free_rolls=2)),
    # chỉ số (khối hiệu ứng áp cho mọi quân trên sân)
    dict(name="Thể Lực", tier=1, group="Chỉ số", text="Cả đội +150 máu.",
         effects=[_stat("hp", "add", [150])]),
    dict(name="Cuồng Phong", tier=1, group="Chỉ số", text="Cả đội +12% tốc đánh.",
         effects=[_stat("as", "mul", [0.12])]),
    dict(name="Tiên Phong", tier=1, group="Chỉ số", text="Quân đứng 2 hàng trước +30 giáp và kháng phép.",
         effects=[dict(when="start", to="self", do="stat", stat="armor", mode="add", values=[30], cond="front_rows"),
                  dict(when="start", to="self", do="stat", stat="mr", mode="add", values=[30], cond="front_rows")]),
    dict(name="Hậu Tập", tier=1, group="Chỉ số", text="Quân đứng 2 hàng sau +20% tốc đánh.",
         effects=[dict(when="start", to="self", do="stat", stat="as", mode="mul", values=[0.20], cond="back_rows")]),
    dict(name="Sức Mạnh Thần", tier=2, group="Chỉ số", text="Cả đội +15% công và +15 sức mạnh phép.",
         effects=[_stat("ad", "mul", [0.15]), _stat("ap", "add", [15])]),
    dict(name="Thành Trì", tier=2, group="Chỉ số", text="Cả đội +25 giáp và kháng phép.",
         effects=[_stat("armor", "add", [25]), _stat("mr", "add", [25])]),
    dict(name="Khát Máu", tier=2, group="Chỉ số", text="Cả đội hút máu 12%.", effects=[_stat("vamp", "add", [0.12])]),
    dict(name="Khởi Đầu Thuận Lợi", tier=2, group="Chỉ số", text="Đầu trận cả đội nhận khiên 200 trong 8 giây.",
         effects=[dict(when="start", to="self", do="shield", values=[200], duration=8)]),
    dict(name="Ban Phước", tier=3, group="Chỉ số", text="Cả đội +300 máu, +20% công, +20 sức mạnh phép.",
         effects=[_stat("hp", "add", [300]), _stat("ad", "mul", [0.20]), _stat("ap", "add", [20])]),
    # đồ
    dict(name="Túi Đồ", tier=1, group="Đồ", text="Nhận 1 mảnh đồ ngẫu nhiên.", econ=dict(components=1)),
    dict(name="Rương Đồ", tier=2, group="Đồ", text="Nhận 2 mảnh đồ ngẫu nhiên.", econ=dict(components=2)),
    dict(name="Kho Vũ Khí", tier=3, group="Đồ", text="Nhận 1 đồ ghép ngẫu nhiên và 1 mảnh đồ.",
         econ=dict(components=1, combined=1)),
]
for _r in ("Hộ vệ", "Đấu sĩ", "Sát thủ", "Xạ thủ", "Pháp sư"):
    AUGMENTS.append(dict(name=f"Ấn {_r}", tier=2, group="Tộc hệ", text=f"Nhận 1 ấn {_r}: gắn lên tướng để tướng đó "
                         f"thêm hệ {_r}.", econ=dict(emblem=_r)))
for _o in ("Hy Lạp", "Bắc Âu", "Ai Cập", "Trung Hoa", "Việt Nam", "Nhật Bản", "Ấn Độ", "Celt", "Quái thú"):
    AUGMENTS.append(dict(name=f"Tim {_o}", tier=1, group="Tộc hệ", text=f"+1 vào số đếm tộc {_o}.",
                         trait_bonus={_o: 1}))
    AUGMENTS.append(dict(name=f"Ấn {_o}", tier=3, group="Tộc hệ", text=f"Nhận 1 ấn {_o} và 5 vàng.",
                         econ=dict(emblem=_o, gold_now=5)))

# ----------------------------------------------------------------------------- quái và vòng đặc biệt
_M = dict(ap=100, crit_chance=0.0, crit_dmg=1.4, mana_start=0, mana_max=0)
MONSTERS = [
    dict(name="Quỷ Nhỏ", stats=dict(_M, hp=350, ad=35, armor=10, mr=10, range=1, **{"as": 0.6})),
    dict(name="Quỷ Ném Đá", stats=dict(_M, hp=280, ad=30, armor=5, mr=5, range=3, **{"as": 0.6})),
    dict(name="Người Đá", stats=dict(_M, hp=1300, ad=70, armor=40, mr=30, range=1, **{"as": 0.5})),
    dict(name="Sói Âm Phủ", stats=dict(_M, hp=900, ad=85, armor=25, mr=25, range=1, **{"as": 0.8})),
    dict(name="Sói Đầu Đàn", stats=dict(_M, hp=1600, ad=120, armor=35, mr=35, range=1, **{"as": 0.8})),
    dict(name="Chim Lửa", stats=dict(_M, hp=1100, ad=110, armor=20, mr=40, range=3, **{"as": 0.7})),
    dict(name="Rồng Già", stats=dict(_M, hp=9000, ad=260, armor=80, mr=80, range=2, **{"as": 0.6})),
    dict(name="Thủy Quái", stats=dict(_M, hp=15000, ad=380, armor=100, mr=100, range=2, **{"as": 0.6})),
]
# vòng quái: (giai đoạn, vòng) -> [(tên quái, sao, ô nửa sân)]
PVE = {
    "1-2": [("Quỷ Nhỏ", 1, 24), ("Quỷ Nhỏ", 1, 26)],
    "1-3": [("Quỷ Nhỏ", 1, 23), ("Quỷ Nhỏ", 1, 25), ("Quỷ Ném Đá", 1, 3)],
    "1-4": [("Quỷ Nhỏ", 1, 23), ("Quỷ Nhỏ", 1, 25), ("Quỷ Ném Đá", 1, 2), ("Quỷ Ném Đá", 1, 4)],
    "2-7": [("Người Đá", 1, 22), ("Người Đá", 1, 24), ("Người Đá", 1, 26)],
    "3-7": [("Sói Âm Phủ", 1, 21), ("Sói Âm Phủ", 1, 23), ("Sói Âm Phủ", 1, 25), ("Sói Âm Phủ", 1, 27),
            ("Sói Đầu Đàn", 1, 17)],
    "4-7": [("Chim Lửa", 1, 8), ("Chim Lửa", 1, 10), ("Chim Lửa", 1, 12), ("Chim Lửa", 1, 2), ("Chim Lửa", 1, 4)],
    "5-7": [("Rồng Già", 1, 17)],
    "6-7": [("Thủy Quái", 1, 17)],
    "7-7": [("Thủy Quái", 1, 17)],
}
GAME_V5 = dict(
    start_hp=100, start_gold=0, stage1_income=3, max_stage=8, bench_slots=9, item_bench=10, max_items=3,
    stage_damage={"1": 0, "2": 2, "3": 5, "4": 8, "5": 10, "6": 12, "7": 17, "8": 20},
    augment_rounds=["2-1", "3-2", "4-2"],
    augment_tier_patterns=[[1, 1, 2], [1, 2, 2], [2, 2, 2], [1, 2, 3], [2, 2, 3], [2, 3, 3], [1, 1, 3]],
    carousel_costs={"1": [1], "2": [1, 2], "3": [2, 3], "4": [3, 4], "5": [3, 4, 5], "6": [3, 4, 5], "7": [4, 5],
                    "8": [4, 5]},
    loot=dict(note="thắng quái nhận đủ phần rơi, thua nhận một nửa (ít nhất 1). Giai đoạn 1 mỗi vòng 1-2 quả "
                   "(55% mảnh đồ, 30% 2-4 vàng, 15% tướng 1 tiền); từ giai đoạn 2: 2 mảnh đồ + 3 vàng, từ giai đoạn 4 "
                   "thêm 1 mảnh. Bảo hiểm: hết giai đoạn 1 ai cũng có ít nhất 2 mảnh.",
              stage1_orbs={"1-2": 1, "1-3": 1, "1-4": 2}, later_components=2, later_gold=3, late_extra_stage=4,
              stage1_min_components=2),
    pve=PVE,
    note=("Giai đoạn 1: 1-1 chọn tướng chung, 1-2..1-4 đánh quái. Từ giai đoạn 2: x-1..x-3 đấu người (lõi ở 2-1, 3-2, "
          "4-2), x-4 chọn tướng chung, x-5, x-6 đấu người, x-7 đánh quái. Máu mất = sát thương giai đoạn + số quân "
          "địch còn sống."),
)


def build() -> dict:
    v4 = V4.build()
    names = [c["name"] for c in v4["champions"]]
    assert set(names) == set(CHAMP_SKILL), set(names) ^ set(CHAMP_SKILL)
    traits = []
    for t in v4["traits"]:
        traits.append(dict(id=t["id"], name=t["name"], kind=t["kind"], breakpoints=t["breakpoints"],
                           scope=t["scope"], effects=_trait_effects(t["name"], t["kind"], t["effects"]),
                           text=t["text"]))
    champs = []
    for c in v4["champions"]:
        rs = ROLE_STATS[c["role"]]
        m = HP_AD_BY_COST[c["cost"]]
        th, td = V4.TWEAK.get(c["name"], (1.0, 1.0))
        key, sname, stext = CHAMP_SKILL[c["name"]]
        stats = dict(hp=round(rs["hp"] * m * th / 5) * 5, ad=round(rs["ad"] * m * td), ap=100,
                     **{"as": rs["as_"]}, armor=rs["armor"] + DEF_PER_COST * (c["cost"] - 1),
                     mr=rs["mr"] + DEF_PER_COST * (c["cost"] - 1), range=rs["range"],
                     crit_chance=0.25, crit_dmg=1.4, mana_start=rs["mana"][0], mana_max=rs["mana"][1])
        if c["role"] == "Pháp sư" and c["cost"] >= 4:
            stats["mana_max"] += 10
        champs.append(dict(id=c["id"], name=c["name"], cost=c["cost"], role=c["role"], traits=c["traits"],
                           stats=stats, skill=dict(template=key, name=sname, text=stext,
                                                   effects=SKILLS[key](c["cost"]))))
    items = []
    for name, eff in COMPONENTS:
        items.append(dict(id=len(items), name=name, kind="component", effects=eff))
    for a, b, name, group, eff in COMBINED:
        items.append(dict(id=len(items), name=name, kind="combined", recipe=[_C[a], _C[b]], group=group,
                          effects=COMPONENTS[_C[a]][1] + COMPONENTS[_C[b]][1] + eff))
    assert len(items) == 44 and len({tuple(sorted(i["recipe"])) for i in items if i["kind"] == "combined"}) == 36
    base = json.load(open(os.path.join(HERE, "catalog_v4_1.json"), encoding="utf-8"))
    out = dict(
        version="catalog-v5-combat-0",
        note=("Luật bản 5, phần giao tranh (08/11/2026): 59 tướng như bản 4 với chỉ số kiểu ĐTCL (máu, công, tốc "
              "đánh, sức mạnh phép, giáp, kháng phép, tầm, năng lượng, chí mạng) và kỹ năng theo giá; 24 tộc hệ; "
              "8 mảnh đồ + 36 đồ ghép có hiệu ứng riêng. Mọi hiệu ứng viết bằng khối hiệu ứng "
              "(battlesim/effects.py), giao tranh do battlesim/sim2.py mô phỏng. Sinh bằng "
              "autochess/data/make_catalog_v5.py."),
        star_mult=STAR_MULT,
        combat_v5=dict(dt=0.1, move_time=0.5, max_time=40.0, overtime_start=20.0, overtime_span=5.0,
                       mana_per_attack=10.0, mana_lock=1.0, mana_from_damage=0.02, mana_from_damage_cap=15.0,
                       wound=0.33, big_hp=1600.0, cast_time=0.3, as_cap=5.0,
                       note="hiệp phụ: từ giây 20 mọi sát thương x(1 + (t-20)/5); hết 40 giây hai bên cùng thua"),
        champions=champs, traits=traits, items=items,
        item_rules=dict(max_per_unit=3, bench_slots=10, remove_and_swap=True,
                        note="tháo / đổi đồ được; tách đồ ghép thành mảnh để mở rộng sau"),
        pool_size_by_cost=base["pool_size_by_cost"], shop_odds=base["shop_odds"], xp_to_next=base["xp_to_next"],
        economy=base["economy"], rules=base["rules"], stage_damage=base["stage_damage"],
        unit_damage_scale=base["unit_damage_scale"],
        augments=[dict(id=i, **a) for i, a in enumerate(AUGMENTS)],
        monsters=[dict(id=len(champs) + i, name=m["name"], cost=0, role="Quái", traits=[], stats=m["stats"],
                       skill=dict(template="none", name="—", text="Quái vật.", effects=[]))
                  for i, m in enumerate(MONSTERS)],
        game_v5=GAME_V5,
    )
    return out


def main():
    out = build()
    path = os.path.join(HERE, "catalog_v5.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"đã ghi {path}: {len(out['champions'])} tướng, {len(out['traits'])} tộc hệ, {len(out['items'])} đồ")


if __name__ == "__main__":
    main()
