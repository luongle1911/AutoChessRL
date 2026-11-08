"""Kiểm thử tính tay được."""
import numpy as np
import pytest

from battlesim import Simulator, make_batch
from battlesim.layout import EV_ATTACK, EV_MOVE

from battlesim.tests.helpers import bounce_counts, champ, tiny_config


@pytest.mark.parametrize("level", [0, 1])
def test_dummy_dies_at_ten_seconds(level):
    # 10 sát thương mỗi giây vào hình nộm 100 máu -> chết đúng ở bước 100 (giây thứ 10)
    cfg = tiny_config(champ("Đánh", ad=10, atk_speed=1.0, rng=8), champ("Hình nộm", hp=100))
    r = Simulator(cfg, level=level).run(make_batch([[(0, 1, 3)]], [[(1, 1, 3)]]))
    assert r.result[0] == 0
    assert r.death_step[0, 10] == 100
    assert r.end_step[0] == 100


def test_armor_halves_damage():
    cfg = tiny_config(champ("Đánh", ad=10, atk_speed=1.0, rng=8),
                      champ("Hình nộm giáp", hp=100, armor=100))
    r = Simulator(cfg).run(make_batch([[(0, 1, 3)]], [[(1, 1, 3)]]))
    assert r.death_step[0, 10] == 200


def test_expected_crit_is_deterministic_multiplier():
    # 25% chí mạng ×1.4 -> hệ số kỳ vọng 1.1 -> 11 sát thương mỗi đòn
    cfg = tiny_config(champ("Đánh", ad=10, atk_speed=1.0, rng=8, crit=0.25, crit_mult=1.4),
                      champ("Hình nộm", hp=1000))
    r = Simulator(cfg).run(make_batch([[(0, 1, 3)]], [[(1, 1, 3)]]))
    assert r.result[0] == 2                  # hết giờ: 30 đòn × 11 = 330 < 1000
    assert r.dmg_dealt[0, 0] == pytest.approx(30 * 11.0, rel=1e-5)


def test_star_multiplier():
    cfg = tiny_config(champ("Đánh", ad=10, atk_speed=1.0, rng=8), champ("Hình nộm", hp=100))
    r = Simulator(cfg).run(make_batch([[(0, 2, 3)]], [[(1, 3, 3)]]))
    # 2 sao: 15 sát thương/đòn; 3 sao: 324 máu -> chết ở đòn thứ 22
    assert r.death_step[0, 10] == 220


def test_ranged_unit_in_range_does_not_move():
    cfg = tiny_config(champ("Cung", ad=10, atk_speed=1.0, rng=4), champ("Hình nộm", hp=100))
    # hàng 3 của phe 0 và hàng 3 (đã xoay) của phe 1 cách nhau 1 ô
    r = Simulator(cfg, log=True).run(make_batch([[(0, 1, 24)]], [[(1, 1, 24)]]))
    ev = r.events[0]
    assert not np.any(ev[:, 1] == EV_MOVE)
    assert np.sum(ev[:, 1] == EV_ATTACK) == 10


def test_melee_walks_to_target_then_attacks():
    cfg = tiny_config(champ("Kiếm", ad=10, atk_speed=1.0, rng=1), champ("Hình nộm", hp=100))
    # phe 0 ở hàng sau (0), hình nộm ở hàng sau phe 1 (hàng 7): phải đi
    r = Simulator(cfg, log=True).run(make_batch([[(0, 1, 3)]], [[(1, 1, 3)]]))
    ev = r.events[0]
    moves = ev[ev[:, 1] == EV_MOVE]
    attacks = ev[ev[:, 1] == EV_ATTACK]
    assert len(moves) > 0 and len(attacks) == 10
    assert moves[:, 0].max() < attacks[:, 0].min()   # đi xong mới đánh
    assert r.result[0] == 0


def test_empty_side_ends_immediately():
    cfg = tiny_config(champ("Đánh", ad=10, atk_speed=1.0))
    r = Simulator(cfg).run(make_batch([[(0, 1, 3)]], [[]]))
    assert r.result[0] == 0 and r.end_step[0] == 1


def test_invalid_input_rejected():
    cfg = tiny_config(champ("A"), champ("B"))
    sim = Simulator(cfg)
    with pytest.raises(ValueError):
        sim.run(make_batch([[(0, 1, 3), (1, 1, 3)]], [[(0, 1, 3)]]))   # trùng ô
    with pytest.raises(ValueError):
        sim.run(make_batch([[(5, 1, 3)]], [[(0, 1, 3)]]))               # mã tướng sai
    with pytest.raises(ValueError):
        sim.run(make_batch([[(0, 4, 3)]], [[(0, 1, 3)]]))               # số sao sai


def test_no_deadlock_when_two_allies_want_same_cell(cfg):
    """Hồi quy: Cung Thủ và Mục Sư cùng muốn ô 22, cùng khoảng cách -> trước đây chờ nhau mãi."""
    from battlesim.viewer import parse_team
    ta = parse_team("Khiên Sắt*2@24, Kiếm Khách@23, Thích Khách@25, Cung Thủ*2@3, Pháp Sư@4, Mục Sư@2", cfg)
    tb = parse_team("Khiên Sắt@24, Kiếm Khách*2@22, Thích Khách@26, Cung Thủ@3, Pháp Sư*2@1, Mục Sư@5", cfg)
    r = Simulator(cfg).run(make_batch([ta], [tb], seeds=[1]))
    assert r.end_step[0] < cfg.max_steps


USER_A = "Thích Khách*2@7, Pháp Sư*3@15, Mục Sư*2@12, Pháp Sư*2@8, Thích Khách*1@1, Mục Sư*3@19"
USER_B = "Khiên Sắt*2@21, Khiên Sắt*1@9, Cung Thủ*2@14, Cung Thủ*1@0, Khiên Sắt*1@27, Khiên Sắt*3@26"


def test_assassin_walks_around_allies(cfg):
    """Hồi quy (trận của người dùng 04/11): Thích Khách 1 sao kẹt sau hai đồng đội, đi qua lại
    giữa hai ô cùng khoảng cách hơn 100 bước thay vì lùi một ô rồi vòng ra sau."""
    from battlesim.viewer import parse_team
    from battlesim.hexgrid import half_to_board
    sim = Simulator(cfg, log=True)
    ta, tb = parse_team(USER_A, cfg), parse_team(USER_B, cfg)
    for seed in (0, 1):
        r = sim.run(make_batch([ta], [tb], seeds=[seed]))
        ev = r.events[0]
        assert max(bounce_counts(ev)) < 3   # code cũ: 13
        # Thích Khách 1 sao (ô tướng 4) phải ra đòn trong 10 giây đầu
        first_attack = ev[(ev[:, 1] == EV_ATTACK) & (ev[:, 2] == 4)][:, 0]
        assert len(first_attack) > 0 and first_attack.min() <= 100
