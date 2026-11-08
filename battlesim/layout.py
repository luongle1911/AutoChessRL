"""Chỉ số cột cho các mảng trạng thái.

Trạng thái động được gói thành hai mảng:
  F: (B, U, NF) float32  — các đại lượng thực
  I: (B, U, NI) int16    — các đại lượng nguyên
Mỗi cột là một thuộc tính; gói lại để hàm Numba không phải nhận hàng chục tham số.
Các hằng số dưới đây là int toàn cục nên Numba coi là hằng lúc biên dịch.
"""

U = 20            # số ô tướng mỗi trận (10 mỗi bên)
U_PER_SIDE = 10
N_CELLS = 56
N_HALF = 28

# ---- Bảng tĩnh của tướng (cfg.stats, float32) ----
S_HP = 0
S_AD = 1
S_AS = 2
S_ARMOR = 3
S_MR = 4
S_CRIT_CH = 5
S_CRIT_MULT = 6
S_MANA_MAX = 7
S_MANA_START = 8
S_SK_POWER = 9
S_SK_PARAM = 10
N_STATS = 11
STAT_KEYS = ("hp", "ad", "as", "armor", "mr", "crit_chance", "crit_mult",
             "mana_max", "mana_start", "skill_power", "skill_param")

# ---- F: trạng thái thực ----
F_HP = 0
F_HP_MAX = 1
F_AD = 2          # công hiệu dụng (đã gồm chí mạng kỳ vọng ở chế độ tất định)
F_CRIT_CH = 3
F_CRIT_MULT = 4
F_AS = 5
F_ARMOR = 6
F_MR = 7
F_MANA = 8
F_MANA_MAX = 9
F_ATK_CD = 10     # giây còn lại tới đòn đánh kế tiếp
F_MOVE_CD = 11    # số bước còn lại của lần di chuyển hiện tại
F_STUN = 12       # giây choáng còn lại (nấc 2)
F_SHIELD = 13     # lượng khiên (nấc 2)
F_PEND = 14       # sát thương chờ áp trong bước này
F_DEALT = 15      # thống kê: sát thương gây ra
F_TAKEN = 16      # thống kê: sát thương gánh chịu
NF = 17

# ---- I: trạng thái nguyên ----
I_CHAMP = 0       # mã tướng, -1 = ô trống
I_STAR = 1
I_SIDE = 2
I_ORD = 3         # thứ tự trong đội; dùng phá hòa để kết quả không phụ thuộc chỉ số mảng
I_CELL = 4
I_TARGET = 5
I_RANGE = 6
I_ALIVE = 7
I_BLOCKED = 8     # số bước liên tiếp muốn đi mà không đi được
I_DEATH = 9       # bước chết, -1 = còn sống
I_CASTS = 10      # số lần tung chiêu (nấc 2)
I_AVOID = 11      # ô vừa hòa tuyệt đối khi tranh, tránh ở bước kế tiếp; -1 = không
I_GOAL = 12       # chỗ đứng đánh đã cam kết đi tới; -1 = chưa có
I_PREV = 13       # ô vừa rời ở lần đi gần nhất; không quay lại ngay trừ khi đã chờ đủ lâu
NI = 14

# ---- Nhật ký sự kiện: ev (B, MAX_EVENTS, 6) float32 ----
# cột: bước, loại, nguồn, đích, giá trị, ô
EV_MOVE = 0
EV_ATTACK = 1
EV_DEATH = 2
EV_CAST = 3

# ---- Kết quả trận ----
RES_SIDE0 = 0
RES_SIDE1 = 1
RES_BOTH_LOSE = 2   # hết giờ hoặc hai bên chết cùng lúc: cả hai cùng thua
