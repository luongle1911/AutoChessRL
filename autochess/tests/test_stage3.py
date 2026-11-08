"""Giai đoạn 3: quan sát bản 3 (hàng chờ đối thủ), nạp mạng bản 2 vào mạng bản 3 (cột mới = 0),
đệm ván 4 người lên 7 đối thủ, sảnh nhiều người (ghế agent, đối thủ gộp, kho theo cặp), đánh giá 8 người."""
import numpy as np
import pytest

from autochess.core import reset, step, legal_actions, rng
from autochess.core.combat import FormulaCombat
from autochess.rl.observe import observe, flatten_batch, pad_opponents, KEYS2, KEYS3, OPP_KEYS

from .helpers import CAT, COMBAT, deterministic_actions


def _state(P=8, B=6, T=400, seed=0):
    s = reset(CAT, rng.make_seeds(seed, B), P)
    for t in range(T):
        m = legal_actions(CAT, s)
        s, _ = step(CAT, COMBAT, s, deterministic_actions(s, t, m), mask=m)
    return s


@pytest.mark.parametrize("P", [2, 4, 8])
def test_v3_adds_opponent_bench(P):
    s = _state(P, 8, 300)
    o2 = observe(CAT, s, COMBAT, version=2)
    o3 = observe(CAT, s, COMBAT, version=3)
    assert tuple(o3) == KEYS3
    O = P - 1
    assert o3["opp_bench_id"].shape == (8, P, O, 9)
    # hàng chờ đối thủ đúng là quân ô 0-8 của ghế đó
    u = s.units.astype(np.int64)
    oi = (np.arange(P)[:, None] + 1 + np.arange(O)[None, :]) % P
    exp = np.where(u[..., :9] >= 0, u[..., :9] // 3, -1)[:, oi]
    assert np.array_equal(o3["opp_bench_id"], exp)
    assert (o3["opp_bench_id"] >= 0).any(), "phải có quân trên hàng chờ trong trạng thái thử"
    # các khóa chung giống bản 2, trừ những cột dùng kho nhìn thấy được (trừ thêm hàng chờ đối thủ)
    for k in KEYS2:
        if k in ("units", "shop", "opp_board_pool"):
            continue
        assert np.array_equal(o2[k], o3[k]), k
    assert (o3["units"][..., 8] <= o2["units"][..., 8] + 1e-6).all()


def test_v3_still_hides_opponent_shop():
    s = _state(4, 4, 250)
    a = observe(CAT, s, COMBAT, version=3)
    t = s.copy()
    for b in range(4):
        for k in range(5):
            if t.shop[b, 1, k] >= 0:
                t.pool[b, t.shop[b, 1, k]] += 1
                t.shop[b, 1, k] = -1
    b_ = observe(CAT, t, COMBAT, version=3)
    for k in KEYS3:
        if k != "mask":
            assert np.array_equal(a[k][:, 0], b_[k][:, 0]), k


def test_pad_opponents_looks_like_eliminated_players():
    s = _state(4, 4, 200)
    o = flatten_batch(observe(CAT, s, COMBAT, version=3))
    p = pad_opponents(o, 7)
    for k in OPP_KEYS:
        if k not in o:
            continue
        assert p[k].shape[1] == 7, k
        assert np.array_equal(p[k][:, :3], o[k]), k
    assert not p["opp_alive"][:, 3:].any()
    assert (p["opp_board_id"][:, 3:] == -1).all() and (p["opp_bench_id"][:, 3:] == -1).all()
    assert pad_opponents(o, 3)["opp"] is o["opp"]


torch = pytest.importorskip("torch")


def _v2_and_v3(seed=0):
    from autochess.rl.nets import build_net
    from autochess.rl.config import load_config
    from autochess.rl.compat import load_into
    cfg = load_config()
    torch.manual_seed(seed)
    v2 = build_net(CAT, cfg["net"], 8)
    cfg3 = dict(cfg["net"], opp_bench=True)
    v3 = build_net(CAT, cfg3, 8)
    rep = load_into(v3, v2.state_dict())
    return v2, v3, rep


def test_v2_checkpoint_loads_into_v3_and_plays_identically_without_bench():
    from autochess.rl.policy import to_tensors
    v2, v3, rep = _v2_and_v3()
    assert v3.obs_version == 3 and v2.obs_version == 2
    assert rep.expanded == {"opp_enc.0.weight": (9 + 4 * 64, 9 + 6 * 64)}
    assert "nới đầu vào" in str(rep)
    s = _state(8, 6, 400)
    # bỏ hết quân trên hàng chờ (về kho) -> hai quan sát trùng nhau ở mọi cột chung
    t = s.copy()
    for b in range(t.B):
        for p in range(t.P):
            for j in range(9):
                if t.units[b, p, j] >= 0:
                    t.units[b, p, j] = -1
    o2 = to_tensors(flatten_batch(observe(CAT, t, COMBAT, version=2)))
    o3 = to_tensors(flatten_batch(observe(CAT, t, COMBAT, version=3)))
    with torch.no_grad():
        l2, v2_ = v2(o2)
        l3, v3_ = v3(o3)
    assert torch.allclose(l2, l3, atol=1e-5) and torch.allclose(v2_, v3_, atol=1e-5)
    # có quân trên hàng chờ: vẫn y hệt khi cột mới = 0 (chỉ khác ở cột kho nhìn thấy được)
    o3b = to_tensors(flatten_batch(observe(CAT, s, COMBAT, version=3)))
    o3b["units"][..., 8] = to_tensors(flatten_batch(observe(CAT, s, COMBAT, version=2)))["units"][..., 8]
    o2b = to_tensors(flatten_batch(observe(CAT, s, COMBAT, version=2)))
    for k in ("shop", "opp_board_pool"):
        o3b[k] = o2b[k]
    with torch.no_grad():
        assert torch.allclose(v2(o2b)[0], v3(o3b)[0], atol=1e-5)
    # học xong thì hàng chờ có ảnh hưởng
    with torch.no_grad():
        v3.opp_enc[0].weight[:, -128:].normal_()
        assert not torch.allclose(v2(o2b)[0], v3(o3b)[0], atol=1e-3)


def test_v3_checkpoint_roundtrip_and_v2_opponent(tmp_path):
    import os
    from autochess.rl.compat import load_compatible
    from autochess.rl.config import load_config
    from autochess.rl.opponents import PolicyOpponent
    from autochess.rl.policy import Policy
    v2, v3, _ = _v2_and_v3()
    cfg = load_config(sets=["net.opp_bench=true", "num_players=8"])
    path = os.path.join(tmp_path, "v3.pt")
    torch.save(dict(net=v3.state_dict(), config=cfg), path)
    net, c, rep = load_compatible(path, CAT)
    assert net.obs_version == 3 and not rep.expanded
    # đối thủ bản 2 và bản 3 cùng ngồi một sảnh, mỗi bên tự quan sát theo bản của mình
    s = _state(8, 4, 200)
    m = legal_actions(CAT, s)
    for pol in (Policy(v2), Policy(net)):
        opp = PolicyOpponent("x", pol, CAT, COMBAT)
        a = opp.act(s, m, np.array([0, 3, 5, 7]))
        assert a.shape == (4,) and m[np.arange(4), [0, 3, 5, 7], a].all()


def test_rollout_multi_seat_lobbies_and_pairwise_pool():
    from autochess.rl.opponents import OpponentPool, make_opponent
    from autochess.rl.rollout import Rollout, RandomPolicy
    pool = OpponentPool([make_opponent(CAT, COMBAT, n) for n in ("tempo", "fast9", "random")])
    ro = Rollout(CAT, COMBAT, pool, num_games=6, num_players=8, steps=400, seed=3, selfplay_prob=0.3,
                 agent_seats=(1, 2))
    assert ro.S == 8
    n_agent = (ro.agent_seats >= 0).sum(1)
    owned = (ro.owner >= 0).sum(1)
    assert ((n_agent == 8) | ((n_agent >= 1) & (n_agent <= 2))).all()
    assert (n_agent + owned == 8).all()
    out = ro.collect(RandomPolicy())
    fin = out["finished"]
    assert fin and all(f["players"] == 8 for f in fin)
    # mỗi ván không tự đấu: số cặp (agent, đối thủ) được ghi vào kho
    pairs = sum(len(f["agent_placement"]) * len(f["opponents"]) for f in fin)
    assert pool.games.sum() == pairs
    assert out["buf"]["active"].shape == (400, 6, 8)


def test_lobbies_train_8p_and_4p_from_v2_checkpoint(tmp_path):
    import os
    from autochess.rl import train
    from autochess.rl.config import load_config
    from autochess.rl.nets import build_net
    cfg = load_config()
    cfg["versions"] = dict(catalog=CAT.version, combat=COMBAT.version)
    seed = os.path.join(tmp_path, "v2.pt")
    torch.save(dict(net=build_net(CAT, cfg["net"], 2).state_dict(), config=cfg), seed)
    args = ["--set", f"run_dir={tmp_path}/runs", "name=p3", f"init={seed}", "num_players=8",
            "net.opp_bench=true", "env.mix={4: 0.34}", "env.num_games=6", "env.steps=48",
            "opponents.names=[tempo,bank]", "opponents.selfplay_prob=0.5", "opponents.agent_seats=[1,2]",
            f"league.seeds={{neo: {seed}}}", "league.snapshot_every=1", "league.max_snapshots=2",
            "ppo.minibatch_size=512", "eval.every=0", "save_every=1", "total_decisions=3000"]
    train.main(args)
    run = os.path.join(tmp_path, "runs", os.listdir(os.path.join(tmp_path, "runs"))[0])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert ck["decisions"] >= 3000 and ck["config"]["net"]["opp_bench"]
    assert ck["net"]["opp_enc.0.weight"].shape[1] == 9 + 6 * 64
    import json
    rows = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    assert rows


def test_skip_waiting_keeps_decisions_and_drops_repeated_waits():
    from autochess.rl.opponents import OpponentPool, make_opponent
    from autochess.rl.rollout import Rollout, RandomPolicy
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo")])
    ro = Rollout(CAT, COMBAT, pool, num_games=4, num_players=8, steps=300, seed=5, selfplay_prob=1.0,
                 skip_waiting=True)
    buf = ro.collect(RandomPolicy())["buf"]
    act, dec, key = buf["active"], buf["decision"], buf["key"]
    assert (dec <= key).all() and (key <= act | ~act).all()
    assert (key & act).sum() < 0.8 * act.sum(), "phải bỏ được phần lớn bước chờ"
    # mỗi lượt có đúng một bước chờ được giữ lại sau quyết định cuối
    assert ((key & act & ~dec).sum()) > 0


def test_evaluate_lobbies_with_bots():
    from autochess.rl.evaluate import evaluate_lobbies, format_lobbies
    from autochess.rl.rollout import RandomPolicy
    r = evaluate_lobbies(CAT, COMBAT, RandomPolicy(), {"A": ["tempo", "fast9", "tempo"],
                                                       "B": ["random"] * 7}, games=8, B=8)
    assert set(r["lobbies"]) == {"A", "B"} and r["lobbies"]["B"]["players"] == 8
    assert 1 <= r["lobbies"]["A"]["mean_place"] <= 4
    assert "agent" in r["elo"] and abs(r["elo"]["random"] - 1500) < 1e-6
    assert "sảnh A" in format_lobbies(r)


@pytest.mark.parametrize("P", [2, 5, 8])
def test_critic_info_predicts_the_actual_pairing_and_dice(P):
    """crit_opp tính lúc lập kế hoạch phải khớp đúng cặp đấu và kết quả trận mà lõi bốc ở cuối lượt."""
    from autochess.core.actions import END_TURN
    s = _state(P, 16, 250, seed=7)
    o = observe(CAT, s, COMBAT, version=3, critic_info="pairing+dice")
    crit = o["crit_opp"]                                                  # [B,P,O,3]
    win_obs = o["opp"][..., 7]
    oi = (np.arange(P)[:, None] + 1 + np.arange(P - 1)[None, :]) % P
    t = s.copy()
    combat_seen = s.done.copy()                 # ván đã xong: lần giao tranh kế tiếp thuộc ván mới
    for _ in range(40):
        m = legal_actions(CAT, t)
        t, info = step(CAT, COMBAT, t, np.full((t.B, P), END_TURN), mask=m)
        combat_seen |= info["reset"] & ~info["combat"]
        cb = info["combat"] & ~combat_seen
        for b in np.flatnonzero(cb):
            for p in range(P):
                j = info["opp"][b, p]
                if not s.alive[b, p] or j < 0:
                    continue
                k = int(np.flatnonzero(oi[p] == j)[0])
                assert crit[b, p, k, 0] == 1 and crit[b, p].sum(0)[0] == 1
                assert bool(crit[b, p, k, 1]) == bool(info["ghost"][b, p])
                won = t.last_result[b, p, 0] == 1
                assert won == (crit[b, p, k, 2] < info["win_prob"][b, p, j])
                assert abs(win_obs[b, p, k] - info["win_prob"][b, p, j]) < 1e-5 or not s.alive[b, j]
        combat_seen |= info["combat"]
        if combat_seen.all():
            break
    assert (combat_seen & ~s.done).sum() >= 8
    p_only = observe(CAT, s, COMBAT, version=3, critic_info="pairing")["crit_opp"]
    assert np.array_equal(p_only[..., :2], crit[..., :2]) and (p_only[..., 2] == 0).all()


def test_critic_info_never_changes_the_policy():
    from autochess.rl.nets import build_net
    from autochess.rl.config import load_config
    from autochess.rl.policy import to_tensors
    from autochess.rl.compat import load_into
    cfg = load_config(sets=["net.opp_bench=true", "net.critic_info=pairing+dice"])
    torch.manual_seed(3)
    net = build_net(CAT, cfg["net"], 8).eval()
    assert net.critic_info == "pairing+dice"
    s = _state(8, 4, 300)
    o = to_tensors(flatten_batch(observe(CAT, s, COMBAT, version=3, critic_info="pairing+dice")))
    q = dict(o, crit_opp=torch.rand_like(o["crit_opp"]))
    with torch.no_grad():
        net.crit_mlp[-1].weight.normal_()
        l1, v1 = net(o)
        l2, v2 = net(q)
        l3, v3 = net({k: v for k, v in o.items() if k != "crit_opp"})
    assert torch.equal(l1, l2) and torch.equal(l1, l3)
    assert not torch.allclose(v1, v2) and not torch.allclose(v1, v3)
    # nạp mạng chưa có crit_mlp: giữ khởi tạo (lớp cuối = 0) nên giá trị như cũ
    plain = build_net(CAT, dict(cfg["net"], critic_info="none"), 8).eval()
    fresh = build_net(CAT, cfg["net"], 8).eval()
    rep = load_into(fresh, plain.state_dict())
    assert rep.new_modules and "mô-đun mới" in str(rep)
    with torch.no_grad():
        assert torch.allclose(plain(o)[1], fresh(o)[1], atol=1e-6)
        assert torch.equal(plain(o)[0], fresh(o)[0])
    from autochess.rl.nets import critic_parameters
    names = {n for n, p in fresh.named_parameters() if any(p is q for q in critic_parameters(fresh))}
    assert any(n.startswith("crit_mlp.") for n in names)


def test_train_with_critic_info(tmp_path):
    import os
    from autochess.rl import train
    train.main(["--set", f"run_dir={tmp_path}/runs", "name=ci", "num_players=8", "net.opp_bench=true",
                "net.critic_info=pairing+dice", "env.mix={4: 0.5}", "env.num_games=4", "env.steps=48",
                "env.skip_waiting=true", "opponents.names=[tempo]", "opponents.selfplay_prob=0.5",
                "ppo.minibatch_size=256", "eval.every=0", "total_decisions=1500"])
    run = os.path.join(tmp_path, "runs", os.listdir(os.path.join(tmp_path, "runs"))[0])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert any(k.startswith("crit_mlp.") for k in ck["net"])


def _seed_ckpt(tmp_path):
    import os
    from autochess.rl.config import load_config
    from autochess.rl.nets import build_net
    cfg = load_config()
    cfg["versions"] = dict(catalog=CAT.version, combat=COMBAT.version)
    seed = os.path.join(tmp_path, "v2.pt")
    torch.save(dict(net=build_net(CAT, cfg["net"], 2).state_dict(), config=cfg), seed)
    return seed


def test_vec_rollout_with_league(tmp_path):
    """Thu thập 2 tiến trình con: chạy được, kho được đồng bộ (bản chụp mới), kết quả ván vào kho chính."""
    import os
    from autochess.rl import train
    seed = _seed_ckpt(tmp_path)
    train.main(["--set", f"run_dir={tmp_path}/runs", "name=vec", f"init={seed}", "num_players=8", "workers=2",
                "net.opp_bench=true", "env.mix={4: 0.34}", "env.num_games=6", "env.steps=64", "env.skip_waiting=true",
                "opponents.names=[tempo]", "opponents.selfplay_prob=0.3", "opponents.agent_seats=[1,2]",
                f"league.seeds={{neo: {seed}}}", "league.snapshot_every=1", "league.max_snapshots=1",
                "ppo.minibatch_size=512", "eval.every=0", "total_decisions=3000"])
    run = os.path.join(tmp_path, "runs", os.listdir(os.path.join(tmp_path, "runs"))[0])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert ck["decisions"] >= 3000
    assert sum(ck["pool"]["games"]) > 0
    assert any(e["name"].startswith("snap_") for e in ck["league"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cần GPU (chạy trên PC)")
def test_gpu_training_path(tmp_path):
    """Đường chạy GPU: mạng và cập nhật trên cuda, thu thập trên CPU (2 tiến trình), lưu / nạp lại trên CPU."""
    import os
    from autochess.rl import train
    from autochess.rl.compat import load_compatible
    seed = _seed_ckpt(tmp_path)
    train.main(["--set", f"run_dir={tmp_path}/runs", "name=gpu", f"init={seed}", "device=cuda", "num_players=8",
                "workers=2", "net.opp_bench=true", "env.num_games=8", "env.steps=64", "env.skip_waiting=true",
                "opponents.names=[tempo]", "opponents.selfplay_prob=0.5", "ppo.minibatch_size=1024",
                "eval.every=2", "eval.games=8", "eval.B=8", "eval.opponents=[tempo]", "eval.matrix=false",
                "eval.lobbies={A: [tempo, tempo, tempo]}", "eval.lobby_games=8", "total_decisions=4000"])
    run = os.path.join(tmp_path, "runs", os.listdir(os.path.join(tmp_path, "runs"))[0])
    net, cfg, _ = load_compatible(os.path.join(run, "checkpoints", "latest.pt"), CAT, "cpu")
    assert next(net.parameters()).device.type == "cpu"
