"""Sinh catalog_v4_1.json (luật bản 4.1, 08/11/2026): giữ nguyên tướng, chỉ số tướng, tộc hệ và mốc của bản 4,
chỉ đổi con số hiệu ứng của 16 tộc hệ thường để các lối chơi mạnh ngang nhau hơn.

    python3 -m autochess.data.make_catalog_v4_1        # ghi autochess/data/catalog_v4_1.json
    python3 -m autochess.data.balance --catalog autochess/data/catalog_v4_1.json   # kiểm tra

Vì sao: ở bản 4, agent sau 100 lần cập nhật chơi Đấu sĩ ở 94% số ván. Phép thử vòng tròn (data/balance.py:
đội dồn từng tộc hệ đấu nhau ở 3 giai đoạn) cho thấy Đấu sĩ / Quái thú / Lưỡng Hà (đều cộng % máu) mạnh nhất,
Xạ thủ, Hy Lạp, Ai Cập, Celt yếu nhất (thắng TB 0,31-0,72 tùy giai đoạn).
Cách chỉnh: nhân mỗi tộc một hệ số (tự dò: tộc thắng nhiều thì giảm, thắng ít thì tăng, đến khi thắng TB ~0,5),
rồi chỉ đi nửa đường (mũ 0,6) và làm tròn bội số 5, để không đổi chất lối chơi quá mạnh. Kết quả: thắng TB của
từng tộc (gộp 3 giai đoạn) trong khoảng 0,45-0,55; Pháp sư / Xạ thủ vẫn yếu giữa ván mạnh cuối ván, Quái thú
mạnh giữa ván — cố ý giữ, để có lối "mạnh sớm" và lối "mạnh muộn".
Các hệ (class) bị giảm vì tướng nào cũng mang đúng một hệ; các tộc gốc được tăng.
Mạng bản 4 dùng thẳng được (thẻ tộc đọc hiệu ứng từ file luật), nên agent bản 4 tinh chỉnh tiếp được.
"""
from __future__ import annotations

import copy
import json
import os

from . import make_catalog_v4 as V4

HERE = os.path.dirname(os.path.abspath(__file__))

EFFECTS_V41 = {
    "Hộ vệ": [["armor", "flat", [15, 35, 60]], ["mr", "flat", [15, 35, 60]]],     # trước 25/50/90
    "Đấu sĩ": [["hp", "pct", [10, 25, 40, 60]]],                                   # trước 20/40/65/100
    "Sát thủ": [["ad", "pct", [20, 40, 65]]],                                      # trước 25/50/85
    "Xạ thủ": [["ad", "pct", [20, 40, 70]]],                                       # trước 20/45/75
    "Pháp sư": [["ap", "pct", [25, 45, 80]]],                                      # trước 30/60/100
    "Hy Lạp": [["ap", "pct", [15, 40, 75]]],                                       # trước 10/25/45
    "Bắc Âu": [["ad", "pct", [20, 50, 80]]],                                       # trước 15/35/60
    "Ai Cập": [["armor", "flat", [15, 40, 70]], ["mr", "flat", [15, 40, 70]]],     # trước 10/25/45
    "Trung Hoa": [["ad", "pct", [25, 65]], ["ap", "pct", [25, 65]]],               # trước 25/60
    "Việt Nam": [["hp", "pct", [10, 20, 35]]],                                     # trước 8/18/32
    "Nhật Bản": [["armor", "flat", [40, 90]], ["ad", "pct", [15, 30]]],            # trước 30/70 giáp, 10/25
    "Ấn Độ": [["ap", "pct", [15, 30, 50, 80]], ["ad", "pct", [15, 30, 50, 80]]],   # trước 10/20/35/55
    "Lưỡng Hà": [["hp", "pct", [20, 35]]],                                         # trước 25/50
    "Celt": [["mr", "flat", [40, 85]]],                                            # trước 20/45
    "Trung Mỹ": [["ad", "pct", [10, 25]], ["hp", "pct", [5, 10]]],                 # trước 20/50, 10/25
    "Quái thú": [["hp", "pct", [20, 45]], ["armor", "flat", [10, 25]]],            # trước 25/60, 15/35
}


def build() -> dict:
    out = copy.deepcopy(V4.build())
    for t in out["traits"]:
        if t["name"] in EFFECTS_V41:
            t["effects"] = copy.deepcopy(EFFECTS_V41[t["name"]])
    out["version"] = "catalog-v4.1"
    out["note"] = ("Luật bản 4.1 (08/11/2026): như catalog-v4, chỉ đổi con số hiệu ứng của 16 tộc hệ thường để cân "
                   "bằng (Đấu sĩ, Quái thú, Lưỡng Hà, Hộ vệ giảm; Hy Lạp, Ai Cập, Celt, Bắc Âu, Ấn Độ tăng). "
                   "Sinh bằng autochess/data/make_catalog_v4_1.py.")
    return out


def main():
    out = build()
    path = os.path.join(HERE, "catalog_v4_1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"đã ghi {path} ({out['version']})")


if __name__ == "__main__":
    main()
