import numpy as np

from battlesim import hexgrid as hg


def test_distance_is_a_metric():
    d = hg.DIST.astype(int)
    assert np.all(np.diag(d) == 0)
    assert np.array_equal(d, d.T)
    # bất đẳng thức tam giác: d[a,b] <= d[a,c] + d[c,b] với mọi c
    assert np.all(d[:, None, :] <= d[:, :, None] + d[None, :, :])


def test_neighbors_are_distance_one():
    for a in range(56):
        ns = [n for n in hg.NEIGHBORS[a] if n >= 0]
        assert len(set(ns)) == len(ns)
        assert all(hg.DIST[a, n] == 1 for n in ns)
        # đúng các ô ở khoảng cách 1
        assert sorted(ns) == sorted(np.flatnonzero(hg.DIST[a] == 1).tolist())


def test_interior_cell_has_six_neighbors():
    assert np.sum(hg.NEIGHBORS[hg.cell_of(3, 3)] >= 0) == 6


def test_rotation_is_involution_without_fixed_cells():
    r = hg.ROT.astype(int)
    assert np.array_equal(r[r], np.arange(56))
    assert np.all(r != np.arange(56))


def test_rotation_preserves_distance():
    r = hg.ROT.astype(int)
    assert np.array_equal(hg.DIST[np.ix_(r, r)], hg.DIST)


def test_rotation_maps_direction_k_to_k_plus_3():
    r = hg.ROT.astype(int)
    for a in range(56):
        for k in range(6):
            n = hg.NEIGHBORS[a, k]
            m = hg.NEIGHBORS[r[a], (k + 3) % 6]
            assert (n < 0 and m < 0) or (n >= 0 and m == r[n])


def test_mirror_flip_would_be_wrong():
    """Lật gương theo hàng (giữ cột) làm sai khoảng cách — lý do dùng xoay 180°."""
    flip = np.array([hg.cell_of(7 - row, col) for row, col in (hg.row_col(c) for c in range(56))])
    assert not np.array_equal(hg.DIST[np.ix_(flip, flip)], hg.DIST)


def test_half_board_mapping():
    for h in range(28):
        assert hg.row_col(hg.half_to_board(h, 0))[0] == h // 7
        assert hg.row_col(hg.half_to_board(h, 1))[0] == 7 - h // 7
    # hàng đầu hai bên sát nhau: ô giữa hàng đầu của hai phe cách 1 ô
    assert hg.DIST[hg.half_to_board(24, 0), hg.half_to_board(24, 1)] == 1
