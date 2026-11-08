"""Hình học bàn cờ lục giác 8 hàng × 7 cột.

Ô được đánh số cell = hàng * 7 + cột (0–55).
- Phe 0 giữ hàng 0–3, hàng 3 là hàng đầu (sát đối phương).
- Phe 1 giữ hàng 4–7, hàng 4 là hàng đầu.
Kiểu so le "odd-r": hàng lẻ lệch sang phải nửa ô.

Mỗi đội mô tả vị trí trên nửa sân của mình bằng chỉ số 0–27 với cùng quy ước
(hàng 0 = hàng sau, hàng 3 = hàng đầu). Phe 1 được đặt lên bàn bằng phép xoay
180°: (hàng, cột) -> (7 - hàng, 6 - cột). Vì số hàng chẵn nên phép xoay giữ đúng
kiểu so le; lật gương theo hàng thì không.

Tất cả bảng được tính một lần lúc import.
"""
import numpy as np

ROWS = 8
COLS = 7
N_CELLS = ROWS * COLS
N_HALF = N_CELLS // 2

# 6 hướng theo tọa độ trục (q, r), xếp vòng quanh ô.
# Xoay 180° biến hướng k thành hướng (k + 3) % 6.
DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def cell_of(row, col):
    return row * COLS + col


def row_col(cell):
    return cell // COLS, cell % COLS


def to_axial(row, col):
    q = col - (row - (row & 1)) // 2
    return q, row


def from_axial(q, r):
    col = q + (r - (r & 1)) // 2
    return r, col


def _hex_dist(a, b):
    qa, ra = to_axial(*row_col(a))
    qb, rb = to_axial(*row_col(b))
    dq, dr = qa - qb, ra - rb
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _build_tables():
    dist = np.zeros((N_CELLS, N_CELLS), dtype=np.int8)
    for a in range(N_CELLS):
        for b in range(N_CELLS):
            dist[a, b] = _hex_dist(a, b)

    nb = np.full((N_CELLS, 6), -1, dtype=np.int8)
    for a in range(N_CELLS):
        q, r = to_axial(*row_col(a))
        for k, (dq, dr) in enumerate(DIRS):
            row, col = from_axial(q + dq, r + dr)
            if 0 <= row < ROWS and 0 <= col < COLS:
                nb[a, k] = cell_of(row, col)

    rot = np.zeros(N_CELLS, dtype=np.int8)
    for a in range(N_CELLS):
        row, col = row_col(a)
        rot[a] = cell_of(ROWS - 1 - row, COLS - 1 - col)
    return dist, nb, rot


DIST, NEIGHBORS, ROT = _build_tables()


def half_to_board(half_cell, side):
    """Ô trên nửa sân (0–27) của một phe -> ô trên bàn (0–55)."""
    return int(half_cell) if side == 0 else int(ROT[half_cell])


def pixel_center(cell, size=1.0):
    """Tâm ô theo pixel (cho UI): x tăng theo cột, y tăng theo hàng."""
    row, col = row_col(cell)
    w = np.sqrt(3.0) * size
    x = w * (col + 0.5 * (row & 1))
    y = 1.5 * size * row
    return float(x), float(y)
