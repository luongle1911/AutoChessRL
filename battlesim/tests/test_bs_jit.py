"""Biên dịch thử bằng Numba và so khớp với chạy Python thường.
Bỏ qua nếu chưa cài Numba hoặc đang đặt NUMBA_DISABLE_JIT=1."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1",
                                reason="đang tắt JIT")

ROOT = Path(__file__).resolve().parents[2]   # thư mục chứa gói battlesim

SCRIPT = """
import numpy as np, sys
sys.path.insert(0, {root!r})
from battlesim import Simulator, load_config, make_batch
from battlesim.tests.helpers import random_team
cfg = load_config(); rng = np.random.default_rng(12345)
ta = [random_team(rng, cfg.n_champs, 6) for _ in range(40)]
tb = [random_team(rng, cfg.n_champs, 6) for _ in range(40)]
r = Simulator(cfg).run(make_batch(ta, tb))
sys.stdout.write(','.join(map(str, r.result.tolist())))
"""


def test_jit_matches_python(cfg):
    from battlesim import Simulator, make_batch
    from battlesim.tests.helpers import random_team

    rng = np.random.default_rng(12345)
    ta = [random_team(rng, cfg.n_champs, 6) for _ in range(40)]
    tb = [random_team(rng, cfg.n_champs, 6) for _ in range(40)]
    r = Simulator(cfg).run(make_batch(ta, tb))

    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, "-c", SCRIPT.format(root=str(ROOT))],
                         env=env, capture_output=True, text=True, check=True).stdout
    assert out == ",".join(map(str, r.result.tolist()))
