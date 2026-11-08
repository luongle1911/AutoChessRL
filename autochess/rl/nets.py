"""Mạng chính sách + giá trị (PyTorch).

EntityNetV2 (mạng chính, kind = "entity2", đọc quan sát bản 2): như bản 1 nhưng không tham số nào có kích
thước phụ thuộc số tướng N hay số tộc K, nên dùng lại được khi file dữ liệu thay đổi (xem compat.py):
  - thẻ tộc  = MLP(3 mốc, 3 mức thưởng) + phần riêng theo tên tộc
  - thẻ tướng = MLP(giá một-nóng, sức mạnh 1/2/3 sao) + chiếu(TB thẻ các tộc của nó) + phần riêng theo tên
  - tộc là một nhóm thực thể có phiếu chấm riêng (dùng chung cho tộc của mình và của đối thủ)
  - tùy chọn mixer = "attn": vài lớp attention giữa các thực thể trước bước tóm tắt
Phần riêng khởi tạo bằng 0, có phạt nhỏ theo độ lớn (aux_loss), và được lưu cùng tên tướng/tộc.

EntityNet (= EntityNetV1, kind = "entity", quan sát bản 1): mỗi nhóm thực thể (ô quân, ô shop, đối thủ, quân trên sân đối thủ) được chấm
bằng một mạng con DÙNG CHUNG ("phiếu chấm"), rồi tóm tắt bằng trung bình + lớn nhất (không phụ thuộc
thứ tự hay số lượng). Vì vậy cùng một bộ tham số chạy cho P = 2, 4, 8.
Đầu ra "chỉ tay": điểm mua ô i tính từ nhận xét của ô shop i + bức tranh toàn cảnh; điểm bán/chuyển
ô j tính từ nhận xét của ô quân j + toàn cảnh; roll, XP, kết thúc lượt tính từ toàn cảnh.

FlatNet (đối chứng): trải phẳng toàn bộ quan sát, MLP 3 lớp. Chỉ cho một số người chơi cố định.

Tất cả: forward(obs) -> (logits[M, 44] CHƯA che mặt nạ, value[M]); thuộc tính obs_version cho biết
mạng đọc quan sát bản nào.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autochess.core.actions import NUM_ACTIONS
from .observe import (UNIT_F, UNIT_F2, SHOP_F, TRAIT_F, SELF2, OPP_BASE, MEET_COL, WIN_COL, self_dim, opp_dim,
                      UNIT_F4, SHOP_F4, SELF4, OPP4, STAT_UNIT)

OPP_UNIT_F = 6   # sao (3), ô trống, giá/5, sức mạnh/100 (tính trong mạng từ id + sao)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(sizes, act_last: bool = True, out_std: float = np.sqrt(2)) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        layers.append(layer_init(nn.Linear(sizes[i], sizes[i + 1]), std=out_std if last else np.sqrt(2)))
        if not last or act_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def masked_mean_max(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """x[..., K, D], valid[..., K] -> [..., 2D]: trung bình và lớn nhất trên các phần tử có thật.
    Không có phần tử nào thì trả về 0."""
    v = valid.unsqueeze(-1).to(x.dtype)
    cnt = v.sum(-2).clamp(min=1.0)
    mean = (x * v).sum(-2) / cnt
    mx = x.masked_fill(~valid.unsqueeze(-1), -1e9).max(-2).values
    mx = torch.where(valid.any(-1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], -1)


class SlotHead(nn.Module):
    """MLP nhỏ trên [nhận xét ô; toàn cảnh], tách thành hai phép chiếu để khỏi lặp toàn cảnh cho
    từng ô (tương đương toán học với MLP trên phép ghép)."""

    def __init__(self, ent: int, hid: int, out: int, width: int = 64):
        super().__init__()
        self.slot = layer_init(nn.Linear(ent, width))
        self.ctx = layer_init(nn.Linear(hid, width), bias=0.0)
        self.out = layer_init(nn.Linear(width, out), std=0.01)

    def forward(self, h_slot: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.slot(h_slot) + self.ctx(ctx).unsqueeze(-2))
        return self.out(z)


class EntityNet(nn.Module):
    obs_version = 1

    def __init__(self, cat, emb: int = 16, ent: int = 64, hid: int = 256, self_hid: int = 128,
                 separate_critic: bool = False):
        super().__init__()
        N = cat.N
        self.register_buffer("cost", torch.tensor(np.asarray(cat.cost, np.float32) / 5.0))
        self.register_buffer("power", torch.tensor(np.asarray(cat.power, np.float32) / 100.0))
        self.emb = nn.Embedding(N + 1, emb)                     # chỉ số 0 = ô trống
        self.unit_enc = mlp([emb + UNIT_F, ent, ent])
        self.shop_enc = mlp([emb + SHOP_F, ent, ent])
        self.opp_unit_enc = mlp([emb + OPP_UNIT_F, ent, ent])
        self.opp_enc = mlp([opp_dim(cat) + 2 * ent, ent, ent])
        self.self_enc = mlp([self_dim(cat), self_hid, self_hid])
        torso_in = self_hid + 2 * ent + 2 * ent + 3 * ent
        self.torso = mlp([torso_in, hid, hid])
        self.critic_torso = mlp([torso_in, hid, hid]) if separate_critic else None
        self.buy_head = SlotHead(ent, hid, 1)
        self.unit_head = SlotHead(ent, hid, 2)                 # [bán, chuyển]
        self.global_head = layer_init(nn.Linear(hid, 3), std=0.01)   # roll, XP, kết thúc lượt
        self.value_head = layer_init(nn.Linear(hid, 1), std=1.0)

    def _ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids.long() + 1)

    def forward(self, obs: dict):
        # quân của mình
        uid = obs["unit_id"]
        u_valid = uid >= 0
        hu = self.unit_enc(torch.cat([self._ids(uid), obs["units"]], -1))            # [M,18,E]
        pool_u = masked_mean_max(hu, u_valid)
        # shop
        sid = obs["shop_id"]
        s_valid = sid >= 0
        hs = self.shop_enc(torch.cat([self._ids(sid), obs["shop"]], -1))             # [M,5,E]
        pool_s = masked_mean_max(hs, s_valid)
        # quân trên sân đối thủ
        bid = obs["opp_board_id"].long()                                             # [M,O,9]
        b_valid = bid >= 0
        c = bid.clamp(min=0)
        st = obs["opp_board_star"].long()
        vf = b_valid.to(hu.dtype)
        ou = torch.cat([F.one_hot(st, 3).to(hu.dtype) * vf.unsqueeze(-1),
                        (~b_valid).to(hu.dtype).unsqueeze(-1),
                        (self.cost[c] * vf).unsqueeze(-1),
                        (self.power[c, st] * vf).unsqueeze(-1)], -1)
        hb = self.opp_unit_enc(torch.cat([self._ids(bid), ou], -1))                  # [M,O,9,E]
        board_pool = masked_mean_max(hb, b_valid)                                    # [M,O,2E]
        # đối thủ
        opp = obs["opp"]
        ho = self.opp_enc(torch.cat([opp, board_pool], -1))                          # [M,O,E]
        o_valid = obs["opp_alive"].bool()
        meet = opp[..., MEET_COL].unsqueeze(-1)
        pool_o = torch.cat([masked_mean_max(ho, o_valid), (ho * meet).sum(-2)], -1)  # [M,3E]
        # toàn cảnh
        x = torch.cat([self.self_enc(obs["self"]), pool_u, pool_s, pool_o], -1)
        ctx = self.torso(x)
        buy = self.buy_head(hs, ctx).squeeze(-1)                                     # [M,5]
        uh = self.unit_head(hu, ctx)                                                 # [M,18,2]
        logits = torch.cat([buy, uh[..., 0], uh[..., 1], self.global_head(ctx)], -1)
        vctx = ctx if self.critic_torso is None else self.critic_torso(x)
        value = self.value_head(vctx).squeeze(-1)
        assert logits.shape[-1] == NUM_ACTIONS
        return logits, value


EntityNetV1 = EntityNet


class FlatNet(nn.Module):
    """Đối chứng: một-nóng id tướng, trải phẳng tất cả, MLP 3 lớp. Chỉ dùng cho P cố định."""
    obs_version = 1

    def __init__(self, cat, num_players: int = 2, hid: int = 256):
        super().__init__()
        self.N = cat.N
        O = num_players - 1
        n1 = cat.N + 1
        dim = (self_dim(cat) + 18 * (UNIT_F + n1) + 5 * (SHOP_F + n1)
               + O * (opp_dim(cat) + 9 * (n1 + 3)) + O)
        self.body = mlp([dim, hid, hid, hid])
        self.pi = layer_init(nn.Linear(hid, NUM_ACTIONS), std=0.01)
        self.v = layer_init(nn.Linear(hid, 1), std=1.0)

    def forward(self, obs: dict):
        n1 = self.N + 1
        M = obs["self"].shape[0]
        oh = lambda ids: F.one_hot(ids.long() + 1, n1).float()
        parts = [obs["self"],
                 torch.cat([obs["units"], oh(obs["unit_id"])], -1).reshape(M, -1),
                 torch.cat([obs["shop"], oh(obs["shop_id"])], -1).reshape(M, -1),
                 obs["opp"].reshape(M, -1),
                 torch.cat([oh(obs["opp_board_id"]), F.one_hot(obs["opp_board_star"].long(), 3).float()],
                           -1).reshape(M, -1),
                 obs["opp_alive"].float().reshape(M, -1)]
        h = self.body(torch.cat(parts, -1))
        return self.pi(h), self.v(h).squeeze(-1)


class AttnBlock(nn.Module):
    """Một lớp transformer (chuẩn hóa trước) trên tập thực thể, không có mã hóa vị trí: đổi thứ tự thực
    thể thì đầu ra đổi thứ tự theo. Thực thể không hợp lệ không được ai nhìn vào."""

    def __init__(self, d: int, heads: int = 4, ff: int = 2):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = layer_init(nn.Linear(d, 3 * d), std=1.0)
        self.proj = layer_init(nn.Linear(d, d), std=0.1)       # nhỏ: lúc đầu gần như "không trộn"
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(layer_init(nn.Linear(d, ff * d)), nn.ReLU(),
                                layer_init(nn.Linear(ff * d, d), std=0.1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        M, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(M, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / np.sqrt(D // self.h)               # [M,H,T,T]
        att = att.masked_fill(~valid[:, None, None, :], -1e9).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(M, T, D)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))


TRAIT_L4 = 4                       # số mốc tối đa của một tộc mà mạng bản 4 đọc
CHAMP_STAT_NORM = np.array([1000.0, 100.0, 100.0, 50.0, 50.0], np.float32)   # chỉ số gốc 1 sao trong thẻ tướng


def trait_stats_v4(cat) -> np.ndarray:
    """[K, 46] dữ liệu tĩnh của tộc cho thẻ tộc bản 4: 4 mốc/6 (0 = không có mốc), số mốc/4, hiệu ứng cả đội,
    bảng hiệu ứng [4 mốc x 5 chỉ số x (phần trăm, cộng thẳng/50)]."""
    K, L = cat.trait_breakpoints.shape
    if L > TRAIT_L4:
        raise ValueError(f"mạng bản 4 đọc tối đa {TRAIT_L4} mốc mỗi tộc (file dữ liệu có {L})")
    lv = np.asarray(cat.trait_levels, np.int64)
    bp = np.zeros((K, TRAIT_L4), np.float32)
    eff = np.zeros((K, TRAIT_L4, 5, 2), np.float32)
    for k in range(K):
        bp[k, :lv[k]] = cat.trait_breakpoints[k, :lv[k]] / 6.0
        eff[k, :lv[k]] = cat.trait_effects[k, :lv[k]]
    eff[..., 1] /= 50.0
    return np.concatenate([bp, lv[:, None] / float(TRAIT_L4), np.asarray(cat.trait_team, np.float32)[:, None],
                           eff.reshape(K, -1)], 1).astype(np.float32)


def _bp3(cat) -> np.ndarray:
    """3 mốc đầu của mỗi tộc [K,3] (luật bản 4 có tộc 1-4 mốc: mốc thiếu lặp lại mốc cuối). Mạng bản 2 chỉ đọc
    3 mốc; quan sát và mạng cho luật bản 4 sẽ làm lại khi huấn luyện trên luật đó."""
    bp = np.asarray(cat.trait_breakpoints, np.float32)
    lv = np.asarray(cat.trait_levels if cat.trait_levels is not None else np.full(cat.K, bp.shape[1]))
    out = np.zeros((cat.K, 3), np.float32)
    for k in range(cat.K):
        real = bp[k, :lv[k]]
        out[k] = [real[min(i, len(real) - 1)] for i in range(3)]
    return out


class EntityNetV2(nn.Module):
    """opp_bench=True (giai đoạn 3): đọc quan sát bản 3, thấy thêm hàng chờ đối thủ. Quân trên hàng chờ
    dùng chung phiếu chấm quân đối thủ, tóm tắt riêng (TB + lớn nhất) rồi ghép vào CUỐI đầu vào phiếu
    đối thủ (opp_enc). Nạp mạng không có hàng chờ: các cột mới = 0 (EXPANDABLE, xem compat.py).

    Luật bản 4 (catalog có stats): mạng tự chuyển sang quan sát bản 4 (có hàng chờ đối thủ). Thẻ tướng thêm 5
    chỉ số gốc; thẻ tộc đọc bảng hiệu ứng theo mốc (trait_stats_v4); quân đối thủ thêm 5 chỉ số x sao; các
    phiếu chấm quân, shop, bản thân, đối thủ rộng hơn theo quan sát bản 4. Không nạp được từ mạng bản 2/3.

    critic_info = "pairing" | "pairing+dice" (mặc định "none"): critic biết thêm đối thủ thật của vòng này
    (và xúc xắc của trận đó) qua khóa quan sát crit_opp. Chỉ nhánh giá trị đọc nó: giá trị = value_head(toàn
    cảnh) + crit_mlp(toàn cảnh, phiếu đối thủ thật, ghost, xúc xắc, xác suất thắng đối thủ đó); lớp cuối của
    crit_mlp khởi tạo 0. Đầu ra chính sách không phụ thuộc crit_opp (có kiểm thử). Quan sát không có
    crit_opp (đối thủ trong kho, đánh giá) thì chỉ dùng value_head."""
    RES_PARAMS = ("champ_res", "trait_res")      # bảng phần riêng: một hàng cho mỗi tên
    EXPANDABLE = ("opp_enc.0.weight",)           # được nới thêm cột đầu vào (khởi tạo 0) khi nạp
    OPTIONAL = ("crit_mlp.",)                    # mô-đun tùy chọn: nạp checkpoint chưa có thì giữ khởi tạo

    def __init__(self, cat, emb: int = 16, ent: int = 64, hid: int = 256, self_hid: int = 128,
                 card_hid: int = 32, separate_critic: bool = False, mixer: str = "none",
                 mixer_layers: int = 2, mixer_heads: int = 4, res_l2: float = 0.01, opp_bench: bool = False,
                 critic_info: str = "none"):
        super().__init__()
        from .observe import CRITIC_INFO
        if critic_info not in CRITIC_INFO:
            raise ValueError(f"critic_info phải là một trong {CRITIC_INFO}")
        self.critic_info = critic_info
        N, K = cat.N, cat.K
        self.v4 = cat.stats is not None
        self.opp_bench = bool(opp_bench) or self.v4
        self.obs_version = 4 if self.v4 else (3 if self.opp_bench else 2)
        self.champ_names, self.trait_names = list(cat.names), list(cat.trait_names)
        self.catalog_version = cat.version
        # ---- bảng tra từ file dữ liệu (không lưu trong state_dict: tính lại từ catalog hiện tại)
        cost_oh = np.eye(5, dtype=np.float32)[np.asarray(cat.cost) - 1]
        cs = [cost_oh, np.asarray(cat.power, np.float32) / 100.0]
        if self.v4:
            cs.append(np.asarray(cat.stats, np.float32) / CHAMP_STAT_NORM)
            mult = np.where(np.arange(5) < 3, np.asarray(cat.star_mult, np.float32)[:, None], 1.0)   # [3,5]
            self.register_buffer("unit_stats_star", torch.tensor(
                np.asarray(cat.stats, np.float32)[:, None, :] * mult[None] / STAT_UNIT), persistent=False)
        self.register_buffer("champ_stats", torch.tensor(np.concatenate(cs, 1)), persistent=False)
        tm = np.asarray(cat.trait_matrix, np.float32)
        self.register_buffer("champ_traits", torch.tensor(tm / tm.sum(1, keepdims=True)), persistent=False)
        ts = trait_stats_v4(cat) if self.v4 else np.concatenate(
            [_bp3(cat) / 6.0, np.asarray(cat.trait_bonus, np.float32)[:, :3] * 2.0], 1)
        self.register_buffer("trait_stats", torch.tensor(ts), persistent=False)
        self.register_buffer("cost", torch.tensor(np.asarray(cat.cost, np.float32) / 5.0), persistent=False)
        self.register_buffer("power", torch.tensor(np.asarray(cat.power, np.float32) / 100.0), persistent=False)
        # ---- thẻ căn cước
        self.trait_card = mlp([self.trait_stats.shape[1], card_hid, emb], act_last=False, out_std=1.0)
        self.champ_card = mlp([self.champ_stats.shape[1], card_hid, emb], act_last=False, out_std=1.0)
        self.champ_from_traits = layer_init(nn.Linear(emb, emb), std=1.0)
        self.champ_res = nn.Parameter(torch.zeros(N, emb))
        self.trait_res = nn.Parameter(torch.zeros(K, emb))
        self.empty_card = nn.Parameter(torch.zeros(emb))
        self.res_l2 = res_l2
        # ---- phiếu chấm
        v4 = self.v4
        self.unit_enc = mlp([emb + (UNIT_F4 if v4 else UNIT_F2), ent, ent])
        self.shop_enc = mlp([emb + (SHOP_F4 if v4 else SHOP_F), ent, ent])
        self.trait_enc = mlp([emb + TRAIT_F, ent, ent])
        self.opp_unit_enc = mlp([emb + OPP_UNIT_F + 1 + (5 if v4 else 0), ent, ent])
        self.opp_enc = mlp([(OPP4 if v4 else OPP_BASE) + (6 if self.opp_bench else 4) * ent, ent, ent])
        self.self_enc = mlp([SELF4 if v4 else SELF2, self_hid, self_hid])
        # ---- trộn giữa các thực thể (tùy chọn)
        self.mixer = mixer
        if mixer == "attn":
            self.self_tok = layer_init(nn.Linear(self_hid, ent))
            self.type_emb = nn.Parameter(torch.zeros(5, ent))          # bản thân, quân, shop, tộc, đối thủ
            self.blocks = nn.ModuleList([AttnBlock(ent, mixer_heads) for _ in range(mixer_layers)])
        elif mixer != "none":
            raise ValueError(f"mixer phải là none hoặc attn, không phải '{mixer}'")
        torso_in = self_hid + 2 * ent + 2 * ent + 2 * ent + 3 * ent
        self.torso = mlp([torso_in, hid, hid])
        self.critic_torso = mlp([torso_in, hid, hid]) if separate_critic else None
        self.buy_head = SlotHead(ent, hid, 1)
        self.unit_head = SlotHead(ent, hid, 2)
        self.global_head = layer_init(nn.Linear(hid, 3), std=0.01)
        self.value_head = layer_init(nn.Linear(hid, 1), std=1.0)
        self.crit_mlp = (mlp([hid + ent + 3, hid // 2, 1], act_last=False, out_std=0.0)
                         if critic_info != "none" else None)

    # ------------------------------------------------------------ tên tướng/tộc đi cùng state_dict
    def get_extra_state(self):
        return dict(champions=list(self.champ_names), traits=list(self.trait_names),
                    catalog=self.catalog_version)

    def set_extra_state(self, state):
        # compat.adapt_state_dict đã sắp lại các hàng phần riêng theo tên hiện tại trước khi nạp
        if state and (list(state["champions"]) != self.champ_names or list(state["traits"]) != self.trait_names):
            raise ValueError("tên tướng/tộc trong checkpoint khác file dữ liệu hiện tại: "
                             "hãy nạp qua autochess.rl.compat.load_compatible")

    def aux_loss(self) -> torch.Tensor:
        """Phạt nhỏ theo độ lớn phần riêng: mạng ưu tiên giải thích bằng chỉ số chung."""
        if not self.res_l2:
            return self.champ_res.new_zeros(())
        return self.res_l2 * (self.champ_res.pow(2).sum(-1).mean() + self.trait_res.pow(2).sum(-1).mean())

    def cards(self):
        """Thẻ tộc [K, emb] và thẻ tướng [N + 1, emb] (hàng 0 = ô trống)."""
        tc = self.trait_card(self.trait_stats) + self.trait_res
        cc = self.champ_card(self.champ_stats) + self.champ_from_traits(self.champ_traits @ tc) + self.champ_res
        return tc, torch.cat([self.empty_card.unsqueeze(0), cc], 0)

    def _opp_units(self, card, ids, star, pool):
        bid = ids.long()
        valid = bid >= 0
        c = bid.clamp(min=0)
        st = star.long()
        vf = valid.to(self.cost.dtype)
        parts = [F.one_hot(st, 3).to(vf.dtype) * vf.unsqueeze(-1),
                 (~valid).to(vf.dtype).unsqueeze(-1),
                 (self.cost[c] * vf).unsqueeze(-1),
                 (self.power[c, st] * vf).unsqueeze(-1),
                 pool.unsqueeze(-1)]
        if self.v4:
            parts.append(self.unit_stats_star[c, st] * vf.unsqueeze(-1))
        ou = torch.cat(parts, -1)
        return self.opp_unit_enc(torch.cat([card(bid), ou], -1)), valid                 # [M,O,9,E]

    def forward(self, obs: dict):
        tc, cc = self.cards()
        card = lambda ids: cc[ids.long() + 1]
        M = obs["self"].shape[0]
        # quân của mình
        uid = obs["unit_id"]
        u_valid = uid >= 0
        hu = self.unit_enc(torch.cat([card(uid), obs["units"]], -1))                   # [M,18,E]
        # shop
        sid = obs["shop_id"]
        s_valid = sid >= 0
        hs = self.shop_enc(torch.cat([card(sid), obs["shop"]], -1))                    # [M,5,E]
        # tộc của mình và của đối thủ: cùng một phiếu
        tr = obs["traits"]
        Kc = tr.shape[-2]
        t_valid = tr[..., 0] > 0
        ht = self.trait_enc(torch.cat([tc.expand(M, Kc, -1), tr], -1))                 # [M,K,E]
        otr = obs["opp_traits"]
        O = otr.shape[1]
        ot_valid = otr[..., 0] > 0
        hot = self.trait_enc(torch.cat([tc.expand(M, O, Kc, -1), otr], -1))            # [M,O,K,E]
        # quân trên sân (và hàng chờ, bản 3) của đối thủ: cùng một phiếu chấm
        hb, b_valid = self._opp_units(card, obs["opp_board_id"], obs["opp_board_star"], obs["opp_board_pool"])
        opp = obs["opp"]
        parts = [opp, masked_mean_max(hb, b_valid), masked_mean_max(hot, ot_valid)]
        if self.opp_bench:
            hbb, bb_valid = self._opp_units(card, obs["opp_bench_id"], obs["opp_bench_star"], obs["opp_bench_pool"])
            parts.append(masked_mean_max(hbb, bb_valid))
        ho = self.opp_enc(torch.cat(parts, -1))
        o_valid = obs["opp_alive"].bool()
        hself = self.self_enc(obs["self"])
        if self.mixer == "attn":
            toks = torch.cat([self.self_tok(hself).unsqueeze(1) + self.type_emb[0],
                              hu + self.type_emb[1], hs + self.type_emb[2], ht + self.type_emb[3],
                              ho + self.type_emb[4]], 1)
            valid = torch.cat([torch.ones_like(u_valid[:, :1]), u_valid, s_valid, t_valid, o_valid], 1)
            for blk in self.blocks:
                toks = blk(toks, valid)
            n_u, n_s = hu.shape[1], hs.shape[1]
            hu = toks[:, 1:1 + n_u]
            hs = toks[:, 1 + n_u:1 + n_u + n_s]
            ht = toks[:, 1 + n_u + n_s:1 + n_u + n_s + Kc]
            ho = toks[:, 1 + n_u + n_s + Kc:]
        meet = opp[..., MEET_COL].unsqueeze(-1)
        pool_o = torch.cat([masked_mean_max(ho, o_valid), (ho * meet).sum(-2)], -1)
        x = torch.cat([hself, masked_mean_max(hu, u_valid), masked_mean_max(hs, s_valid),
                       masked_mean_max(ht, t_valid), pool_o], -1)
        ctx = self.torso(x)
        buy = self.buy_head(hs, ctx).squeeze(-1)
        uh = self.unit_head(hu, ctx)
        logits = torch.cat([buy, uh[..., 0], uh[..., 1], self.global_head(ctx)], -1)
        vctx = ctx if self.critic_torso is None else self.critic_torso(x)
        value = self.value_head(vctx).squeeze(-1)
        if self.crit_mlp is not None and "crit_opp" in obs:
            co = obs["crit_opp"].to(ho.dtype)                                          # [M,O,3]
            real = co[..., 0:1]
            ci = torch.cat([vctx, (ho * real).sum(-2), co[..., 1:3].sum(-2),
                            (opp[..., WIN_COL:WIN_COL + 1] * real).sum(-2)], -1)
            value = value + self.crit_mlp(ci).squeeze(-1)
        assert logits.shape[-1] == NUM_ACTIONS
        return logits, value


CRITIC_PREFIXES = ("value_head.", "critic_torso.", "v.", "crit_mlp.")


def critic_parameters(net: nn.Module) -> list:
    """Tham số chỉ phục vụ critic (dùng khi "khởi động critic" mà không đụng tới chính sách)."""
    return [p for n, p in net.named_parameters() if n.startswith(CRITIC_PREFIXES)]


def obs_version(net) -> int:
    return getattr(net, "obs_version", 1)


def build_net(cat, cfg: dict, num_players: int) -> nn.Module:
    kind = cfg.get("kind", "entity2")
    if kind == "entity2":
        return EntityNetV2(cat, emb=cfg.get("emb", 16), ent=cfg.get("ent", 64), hid=cfg.get("hid", 256),
                           self_hid=cfg.get("self_hid", 128), card_hid=cfg.get("card_hid", 32),
                           separate_critic=cfg.get("separate_critic", False), mixer=cfg.get("mixer", "none"),
                           mixer_layers=cfg.get("mixer_layers", 2), mixer_heads=cfg.get("mixer_heads", 4),
                           res_l2=cfg.get("res_l2", 0.01), opp_bench=cfg.get("opp_bench", False),
                           critic_info=cfg.get("critic_info", "none"))
    if kind == "entity":
        return EntityNet(cat, emb=cfg.get("emb", 16), ent=cfg.get("ent", 64), hid=cfg.get("hid", 256),
                         self_hid=cfg.get("self_hid", 128), separate_critic=cfg.get("separate_critic", False))
    if kind == "flat":
        return FlatNet(cat, num_players, hid=cfg.get("hid", 256))
    raise ValueError(f"không có loại mạng '{kind}'")
