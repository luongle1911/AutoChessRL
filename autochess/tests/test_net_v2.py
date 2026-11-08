"""Mạng bản 2 (cần PyTorch): chạy với mọi số người chơi và mọi file dữ liệu, số tham số không đổi theo N/K,
nạp checkpoint ghép theo tên, đổi thứ tự id không làm đổi đầu ra, khởi động critic không đụng chính sách."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autochess.core import reset, step, legal_actions, rng, load_catalog
from autochess.core.catalog import DATA_DIR
from autochess.core.combat import FormulaCombat
from autochess.rl.observe import observe, flatten_batch, OPP_KEYS
from autochess.rl.nets import EntityNetV2, EntityNetV1, build_net, critic_parameters
from autochess.rl.compat import load_compatible, load_into, adapt_state_dict
from autochess.rl.policy import Policy, to_tensors, load_checkpoint
from autochess.rl.ppo import ppo_update, flatten_rollout
from autochess.rl.gae import compute_gae
from autochess.rl.opponents import OpponentPool, make_opponent
from autochess.rl.rollout import Rollout
from autochess.rl.config import load_config

from .helpers import CAT, COMBAT, deterministic_actions
from .catalog_variants import variant, translate_state


def _state(cat, P=2, B=6, T=300, seed=0):
    combat = FormulaCombat(catalog=cat)
    s = reset(cat, rng.make_seeds(seed, B), P)
    for t in range(T):
        m = legal_actions(cat, s)
        s, _ = step(cat, combat, s, deterministic_actions(s, t, m), mask=m)
    return s, combat


def _obs(cat, P=2, B=6, T=300, seed=0):
    s, combat = _state(cat, P, B, T, seed)
    return flatten_batch(observe(cat, s, combat, version=2))


def _nonres_params(net):
    return sum(p.numel() for n, p in net.named_parameters() if n not in EntityNetV2.RES_PARAMS)


def _randomize_res(net, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in EntityNetV2.RES_PARAMS:
            p = getattr(net, n)
            p.copy_(0.3 * torch.randn(p.shape, generator=g))


@pytest.mark.parametrize("mixer", ["none", "attn"])
def test_v2_runs_for_any_player_count(mixer):
    torch.manual_seed(0)
    net = EntityNetV2(CAT, mixer=mixer)
    assert 100_000 < sum(p.numel() for p in net.parameters()) < 600_000
    for P in (2, 4, 8):
        logits, value = net(to_tensors(_obs(CAT, P)))
        assert logits.shape == (6 * P, 44) and value.shape == (6 * P,)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("mixer", ["none", "attn"])
def test_v2_param_count_independent_of_catalog(mixer):
    counts = set()
    for N, K in ((18, 5), (20, 6), (24, 8)):
        cat = variant(N, K)
        net = EntityNetV2(cat, mixer=mixer)
        counts.add(_nonres_params(net))
        assert net.champ_res.shape[0] == N and net.trait_res.shape[0] == K
        logits, value = net(to_tensors(_obs(cat, 4, B=3)))
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert len(counts) == 1


@pytest.mark.parametrize("mixer", ["none", "attn"])
def test_v2_opponent_permutation_and_shop_swap(mixer):
    torch.manual_seed(1)
    net = EntityNetV2(CAT, mixer=mixer).eval()
    o = _obs(CAT, 4, B=4)
    p = dict(o)
    perm = [2, 0, 1]
    for k in OPP_KEYS:
        if k in o:
            p[k] = o[k][:, perm]
    with torch.no_grad():
        l1, v1 = net(to_tensors(o))
        l2, v2 = net(to_tensors(p))
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)
    sw = [3, 1, 2, 0, 4]
    q = dict(o, shop=o["shop"][:, sw], shop_id=o["shop_id"][:, sw])
    with torch.no_grad():
        l3, v3 = net(to_tensors(q))
    assert torch.allclose(l1[:, sw], l3[:, :5], atol=1e-5)
    assert torch.allclose(l1[:, 5:], l3[:, 5:], atol=1e-5) and torch.allclose(v1, v3, atol=1e-5)


@pytest.mark.parametrize("mixer", ["none", "attn"])
def test_v2_id_order_does_not_matter(mixer):
    """Cùng tướng, cùng tộc nhưng đánh số khác trong file dữ liệu: nạp theo tên thì đầu ra giống hệt."""
    torch.manual_seed(2)
    cat_b = variant(20, 6, permute=True, seed=5)
    assert sorted(cat_b.names) == sorted(CAT.names) and cat_b.names != CAT.names
    assert cat_b.trait_names != CAT.trait_names
    net_a = EntityNetV2(CAT, mixer=mixer).eval()
    _randomize_res(net_a)
    net_b = EntityNetV2(cat_b, mixer=mixer).eval()
    rep = load_into(net_b, net_a.state_dict())
    assert rep.reordered and not rep.added["champions"] and not rep.dropped["traits"]
    s, combat_a = _state(CAT, 4, B=4, T=320)
    sb = translate_state(s, CAT, cat_b)
    combat_b = FormulaCombat(catalog=cat_b)
    oa = flatten_batch(observe(CAT, s, combat_a, version=2))
    ob = flatten_batch(observe(cat_b, sb, combat_b, version=2))
    assert np.array_equal(oa["mask"], ob["mask"])
    with torch.no_grad():
        la, va = net_a(to_tensors(oa))
        lb, vb = net_b(to_tensors(ob))
    assert torch.allclose(la, lb, atol=1e-4) and torch.allclose(va, vb, atol=1e-4)


def test_load_compatible_maps_by_name(tmp_path):
    torch.manual_seed(3)
    net = EntityNetV2(CAT)
    _randomize_res(net, 7)
    cfg = load_config()
    cfg["versions"] = dict(catalog=CAT.version, combat=COMBAT.version)
    path = os.path.join(tmp_path, "ck.pt")
    torch.save(dict(net=net.state_dict(), config=cfg), path)

    patch = load_catalog(f"{DATA_DIR}/catalog_v1b.json")
    net2, cfg2, rep = load_compatible(path, patch)
    assert rep.added["champions"] == ["Băng Hậu", "Kỵ Vương"] and rep.dropped["champions"] == ["Phong Ảnh"]
    assert rep.added["traits"] == ["Kỵ sĩ"] and not rep.dropped["traits"]
    assert any("catalog-v1b" in w for w in rep.warnings)
    for j, n in enumerate(patch.names):
        if n in CAT.names:
            assert torch.equal(net2.champ_res[j], net.champ_res[CAT.names.index(n)])
        else:
            assert torch.count_nonzero(net2.champ_res[j]) == 0
    assert torch.count_nonzero(net2.trait_res[patch.trait_names.index("Kỵ sĩ")]) == 0
    # mọi tham số không gắn với tên được giữ nguyên
    sd1, sd2 = net.state_dict(), net2.state_dict()
    for k, v in sd1.items():
        if k not in EntityNetV2.RES_PARAMS and k != "_extra_state":
            assert torch.equal(v, sd2[k]), k
    logits, value = net2(to_tensors(_obs(patch, 2, B=4)))
    assert torch.isfinite(logits).all()

    # đổi tên: giữ phần riêng
    renamed = dict(net.state_dict())
    renamed["_extra_state"] = dict(renamed["_extra_state"])
    renamed["_extra_state"]["champions"] = [("Lê Kiếm cũ" if n == "Lê Kiếm" else n)
                                            for n in renamed["_extra_state"]["champions"]]
    net3 = EntityNetV2(CAT)
    rep3 = load_into(net3, renamed, renames={"Lê Kiếm cũ": "Lê Kiếm"})
    assert rep3.renamed == {"Lê Kiếm cũ": "Lê Kiếm"} and not rep3.changed
    assert torch.equal(net3.champ_res, net.champ_res)


def test_direct_load_with_other_names_is_refused():
    net = EntityNetV2(CAT)
    other = EntityNetV2(variant(20, 6, permute=True, seed=5))
    with pytest.raises(Exception):
        other.load_state_dict(net.state_dict())


def test_v1_checkpoint_still_loads(tmp_path):
    """Checkpoint cũ (mạng bản 1, cấu hình net.kind = entity) nạp qua load_checkpoint như trước."""
    net = EntityNetV1(CAT)
    cfg = load_config()
    cfg["net"]["kind"] = "entity"
    path = os.path.join(tmp_path, "v1.pt")
    torch.save(dict(net=net.state_dict(), config=cfg), path)
    net2, cfg2 = load_checkpoint(path, CAT)
    assert isinstance(net2, EntityNetV1) and Policy(net2).obs_version == 1
    with pytest.raises(RuntimeError):
        load_compatible(path, load_catalog(f"{DATA_DIR}/catalog_v1b.json"))


def test_value_warmup_keeps_policy_and_lr_schedule():
    from autochess.rl.train import lr_at
    torch.manual_seed(4)
    net = build_net(CAT, {"kind": "entity2"}, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5)
    pool = OpponentPool([make_opponent(CAT, COMBAT, "tempo")])
    ro = Rollout(CAT, COMBAT, pool, num_games=8, num_players=2, steps=40, seed=3)
    pol = Policy(net)
    out = ro.collect(pol)
    assert "traits" in out["buf"]["obs"]                         # bộ đệm dùng quan sát bản 2
    buf = out["buf"]
    T_, B_, S_ = buf["rewards"].shape
    f = lambda x: x.reshape(T_, B_ * S_)
    adv, ret = compute_gae(f(buf["rewards"]), f(buf["values"]), f(buf["dones"]), f(buf["gammas"]),
                           f(buf["lambdas"]), f(buf["active"]), out["last_values"].reshape(-1))
    batch = flatten_rollout(buf, adv.reshape(T_, B_, S_), ret.reshape(T_, B_, S_))
    obs = to_tensors({k: v[:64] for k, v in batch["obs"].items()})
    with torch.no_grad():
        l0, v0 = net(obs)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    cfg = load_config()["ppo"] | {"minibatch_size": 128, "update_epochs": 2, "pg_coef": 0.0}
    ppo_update(net, opt, batch, cfg, np.random.default_rng(0))
    with torch.no_grad():
        l1, v1 = net(obs)
    assert torch.equal(l0, l1)                                    # chính sách y nguyên
    assert not torch.allclose(v0, v1)                             # critic đã học
    crit = {id(p) for p in critic_parameters(net)}
    for n, p in net.named_parameters():
        assert torch.equal(p, before[n]) or id(p) in crit, n

    pc = dict(lr=1e-3, anneal_lr=False, value_warmup=10, lr_warmup=4)
    assert [lr_at(pc, u, 0, 100) for u in (0, 9, 10, 11, 13, 50)] == [1e-3, 1e-3, 2.5e-4, 5e-4, 1e-3, 1e-3]
    assert lr_at(dict(lr=1e-3, anneal_lr=True), 0, 50, 100) == pytest.approx(5e-4)


def test_head_to_head_v1_vs_v2():
    from autochess.rl.evaluate import head_to_head
    r = head_to_head(CAT, COMBAT, Policy(EntityNetV2(CAT)), Policy(EntityNetV1(CAT)), games=8, B=8)
    assert r["games"] >= 8 and 0 <= r["winrate_a"] <= 1


def test_train_with_attn_and_patched_catalog(tmp_path):
    """Huấn luyện ngắn với file dữ liệu gốc, rồi chạy tiếp (resume) với bản cập nhật giả."""
    from autochess.rl import train
    args = ["--set", f"run_dir={tmp_path}", "name=t", "total_decisions=2000", "env.num_games=8",
            "env.steps=32", "ppo.minibatch_size=256", "eval.every=0", "save_every=1",
            "opponents.names=[random,tempo]", "net.mixer=attn", "ppo.lr_warmup=2"]
    train.main(args)
    run = os.path.join(tmp_path, os.listdir(tmp_path)[0])
    train.main(["--resume", run, "--set", "total_decisions=4000", f"catalog={DATA_DIR}/catalog_v1b.json"])
    ck = torch.load(os.path.join(run, "checkpoints", "latest.pt"), weights_only=False)
    assert ck["decisions"] >= 4000 and ck["net"]["champ_res"].shape[0] == 21
