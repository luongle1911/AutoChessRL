"""Phép thử cân bằng nhanh cho luật có chỉ số (bản 4+): với mỗi tộc hệ thường, dựng một đội "dồn tộc đó"
rồi cho các đội đấu vòng tròn bằng chính công thức giao tranh của luật.

    python3 -m autochess.data.balance --catalog autochess/data/catalog_v4_1.json

Cách dựng đội (ở mỗi giai đoạn: cấp = số quân, giá tối đa, số sao): lấy các thành viên mạnh nhất của tộc
(tối đa đủ mốc cao nhất đạt được), phần còn lại lấp bằng quân làm độ mạnh đội tăng nhiều nhất (đúng cách bot
luật chấm quân). Thô: chưa tính 3 sao, chưa tính tranh tướng trong kho chung, chưa tính kinh tế.
Kết quả: xác suất thắng trung bình của từng đội trước các đội còn lại, và trận khó nhất của nó.
"""
from __future__ import annotations

import argparse

import numpy as np

from autochess.core import load_catalog, FormulaCombat
from autochess.core.combat import board_rating

STAGES = {"giữa (cấp 7, ≤3 tiền, 2 sao)": (7, 3, 1), "cuối (cấp 8, ≤4 tiền, 2 sao)": (8, 4, 1),
          "rất muộn (cấp 9, ≤5 tiền, 2 sao)": (9, 5, 1)}


def trait_board(cat, k, n, maxcost, star):
    pool = [c for c in range(cat.N) if cat.cost[c] <= maxcost]
    mem = sorted((c for c in pool if cat.trait_matrix[c, k]), key=lambda c: -cat.power[c, star])
    bp = [b for b in cat.trait_breakpoints[k, :cat.trait_levels[k]] if b <= min(n, len(mem))]
    take = bp[-1] if bp else 0
    b = [c * 3 + star for c in mem[:take]]
    while len(b) < n:
        cand = [c for c in pool if c * 3 + star not in b]
        boards = np.full((len(cand), 9), -1)
        boards[:, :len(b)] = b
        boards[:, len(b)] = np.array(cand) * 3 + star
        b.append(cand[int(board_rating(cat, boards).argmax())] * 3 + star)
    return np.array(b + [-1] * (9 - n)), take


class _S:
    def __init__(self, B, rnd=25):
        self.units = np.concatenate([np.full((len(B), 9), -1), B], -1)[None].astype(np.int16)
        self.round = np.array([rnd])


def round_robin(cat, stage):
    n, maxcost, star = stage
    kinds = [t.get("kind") for t in sorted(cat.raw["traits"], key=lambda t: t["id"])]
    ks = [k for k in range(cat.K) if kinds[k] != "unique"]
    boards, takes = zip(*(trait_board(cat, k, n, maxcost, star) for k in ks))
    B = np.stack(boards)
    win, _ = FormulaCombat(catalog=cat).evaluate(cat, _S(B))
    w = win[0]
    avg = (w.sum(1) - 0.5) / (len(ks) - 1)
    return [dict(trait=cat.trait_names[k], members=int(t), win=float(a), worst=float(w[i].min()),
                 worst_vs=cat.trait_names[ks[int(w[i].argmin())]], rating=float(board_rating(cat, B[i])))
            for i, (k, t, a) in enumerate(zip(ks, takes, avg))]


def report(cat) -> float:
    spread = 0.0
    for name, st in STAGES.items():
        rows = sorted(round_robin(cat, st), key=lambda r: -r["win"])
        wins = np.array([r["win"] for r in rows])
        spread = max(spread, wins.max() - wins.min())
        print(f"\n{name}: thắng TB từ {wins.min():.2f} đến {wins.max():.2f} (độ lệch chuẩn {wins.std():.3f})")
        for r in rows:
            print(f"  {r['trait']:10s} {r['members']} thành viên  thắng TB {r['win']:.2f}  "
                  f"khó nhất: {r['worst']:.2f} trước {r['worst_vs']}")
    return spread


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="autochess/data/catalog_v4.json")
    a = ap.parse_args()
    c = load_catalog(a.catalog)
    print(c.version)
    report(c)
