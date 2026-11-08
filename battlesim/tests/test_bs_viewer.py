import json

import numpy as np
import pytest

from battlesim import Simulator, make_batch
from battlesim.viewer import parse_team, record, write_html
from battlesim.tests.helpers import random_team


def test_record_matches_full_run(cfg, np_rng):
    """Chạy từng bước để ghi hình phải ra đúng kết quả như chạy trọn trận."""
    for _ in range(10):
        ta = random_team(np_rng, cfg.n_champs, 6)
        tb = random_team(np_rng, cfg.n_champs, 6)
        rep = record(cfg, ta, tb, seed=3)
        r = Simulator(cfg).run(make_batch([ta], [tb], seeds=[3]))
        assert rep["meta"]["result"] == r.result[0]
        assert rep["meta"]["end_step"] == r.end_step[0]
        assert np.allclose(rep["frames"][-1]["dealt"], r.dmg_dealt[0], atol=0.1)
        assert len(rep["frames"]) == rep["meta"]["end_step"] + 1
        assert rep["warnings"] == []


def test_write_html_embeds_data(cfg, np_rng, tmp_path):
    rep = record(cfg, random_team(np_rng, cfg.n_champs, 4), random_team(np_rng, cfg.n_champs, 4))
    path = write_html(rep, tmp_path / "x.html")
    html = path.read_text(encoding="utf-8")
    start = html.index('id="data">') + len('id="data">')
    end = html.index("</script>", start)
    data = json.loads(html[start:end].replace("<\\/", "</"))
    assert data["meta"]["end_step"] == rep["meta"]["end_step"]


def test_parse_team(cfg):
    assert parse_team("Khiên Sắt*2@24, Cung Thủ@3", cfg) == [(0, 2, 24), (2, 1, 3)]
    assert parse_team("1*3@0", cfg) == [(1, 3, 0)]
    with pytest.raises(ValueError):
        parse_team("Khiên Sắt", cfg)
