"""Nạp checkpoint khi file dữ liệu (tướng, tộc, chỉ số) đã thay đổi so với lúc huấn luyện.

    net, cfg, report = load_compatible("runs/<tên>/checkpoints/latest.pt", catalog)
    print(report)

Mạng bản 2 (EntityNetV2) chỉ có hai bảng tham số gắn với từng tướng/tộc: phần riêng champ_res [N, emb]
và trait_res [K, emb]. state_dict lưu kèm tên tướng/tộc lúc huấn luyện (khóa "_extra_state"), nên khi
nạp ta ghép theo TÊN:
  - tên có ở cả hai bên: giữ hàng phần riêng cũ (kể cả khi số thứ tự đổi);
  - tên mới: phần riêng = 0 (mạng đánh giá nó hoàn toàn qua chỉ số);
  - tên đã mất: bỏ hàng đó.
renames = {"tên cũ": "tên mới"} để giữ phần riêng khi một tướng/tộc chỉ đổi tên.
Mọi tham số khác có kích thước không phụ thuộc N, K nên nạp thẳng.

Nới đầu vào (giai đoạn 3): mạng có hàng chờ đối thủ (net.opp_bench) có thêm cột ở cuối đầu vào phiếu đối
thủ. Nạp mạng chưa có: các cột mới = 0, nên lúc đầu mạng chơi y như cũ (chỉ cho các khóa trong
net.EXPANDABLE).

Mạng bản 1 và mạng phẳng phụ thuộc N, K: chỉ nạp được khi tên tướng/tộc giống hệt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import DEFAULTS, deep_merge


@dataclass
class CompatReport:
    kind: str = ""
    kept: dict = field(default_factory=dict)       # {"champions": [...], "traits": [...]}
    added: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)
    renamed: dict = field(default_factory=dict)
    reordered: bool = False
    expanded: dict = field(default_factory=dict)   # {khóa: (số cột cũ, số cột mới)}
    new_modules: list = field(default_factory=list)  # tham số của mô-đun tùy chọn chưa có trong checkpoint
    warnings: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True nếu tập tên hoặc thứ tự id khác lúc huấn luyện (kích thước/ý nghĩa hàng phần riêng đổi)."""
        return self.reordered or any(self.added.get(k) or self.dropped.get(k) for k in ("champions", "traits"))

    def __str__(self) -> str:
        lines = [f"nạp mạng {self.kind}:"]
        for k, label in (("champions", "tướng"), ("traits", "tộc")):
            if k not in self.kept:
                continue
            s = f"  {label}: giữ {len(self.kept[k])}"
            if self.added.get(k):
                s += f", mới {len(self.added[k])} ({', '.join(self.added[k])})"
            if self.dropped.get(k):
                s += f", bỏ {len(self.dropped[k])} ({', '.join(self.dropped[k])})"
            lines.append(s)
        if self.renamed:
            lines.append("  đổi tên: " + ", ".join(f"{a} -> {b}" for a, b in self.renamed.items()))
        if self.reordered:
            lines.append("  thứ tự id khác lúc huấn luyện (đã ghép theo tên)")
        for k, (a, b) in self.expanded.items():
            lines.append(f"  nới đầu vào {k}: {a} -> {b} cột (cột mới = 0)")
        if self.new_modules:
            lines.append(f"  mô-đun mới (giữ khởi tạo): {', '.join(sorted({k.split('.')[0] for k in self.new_modules}))}")
        lines += [f"  CẢNH BÁO: {w}" for w in self.warnings]
        return "\n".join(lines)


def _remap_rows(old: torch.Tensor, old_names: list, new_names: list, renames: dict, rep: CompatReport,
                key: str) -> torch.Tensor:
    mapped = [renames.get(n, n) for n in old_names]
    pos = {n: i for i, n in enumerate(mapped)}
    out = old.new_zeros((len(new_names),) + tuple(old.shape[1:]))
    kept, added = [], []
    for j, n in enumerate(new_names):
        if n in pos:
            out[j] = old[pos[n]]
            kept.append(n)
        else:
            added.append(n)
    rep.kept[key], rep.added[key] = kept, added
    rep.dropped[key] = [o for o, m in zip(old_names, mapped) if m not in set(new_names)]
    for o, m in zip(old_names, mapped):
        if o != m and m in pos and m in new_names:
            rep.renamed[o] = m
    if [n for n in mapped if n in set(new_names)] != [n for n in new_names if n in pos]:
        rep.reordered = True
    return out


def adapt_state_dict(sd: dict, net, renames: dict | None = None) -> tuple[dict, CompatReport]:
    """Sắp lại state_dict `sd` (lưu từ một mạng cùng loại) cho vừa `net` (dựng theo file dữ liệu hiện tại)."""
    renames = renames or {}
    rep = CompatReport(kind=type(net).__name__)
    sd = dict(sd)
    if not hasattr(net, "RES_PARAMS"):
        return sd, rep                         # bản 1 / phẳng: nạp thẳng, load_state_dict sẽ báo nếu lệch
    meta = sd.get("_extra_state")
    if meta is None:
        raise ValueError("checkpoint không có tên tướng/tộc (_extra_state): không phải mạng bản 2?")
    sd["champ_res"] = _remap_rows(sd["champ_res"], list(meta["champions"]), net.champ_names, renames, rep,
                                  "champions")
    sd["trait_res"] = _remap_rows(sd["trait_res"], list(meta["traits"]), net.trait_names, renames, rep, "traits")
    if meta.get("catalog") != net.catalog_version:
        rep.warnings.append(f"file dữ liệu khác lúc huấn luyện: {meta.get('catalog')} -> {net.catalog_version}")
    sd["_extra_state"] = net.get_extra_state()
    own = net.state_dict()
    for k in getattr(net, "EXPANDABLE", ()):
        if k in sd and k in own and sd[k].dim() == 2 and sd[k].shape[0] == own[k].shape[0] \
                and sd[k].shape[1] < own[k].shape[1]:
            w = sd[k].new_zeros(own[k].shape)
            w[:, :sd[k].shape[1]] = sd[k]
            rep.expanded[k] = (int(sd[k].shape[1]), int(own[k].shape[1]))
            sd[k] = w
    opt = tuple(getattr(net, "OPTIONAL", ()))
    for k in own:
        if opt and k.startswith(opt) and k not in sd:
            sd[k] = own[k]
            rep.new_modules.append(k)
    return sd, rep


def load_into(net, sd: dict, renames: dict | None = None) -> CompatReport:
    """Nạp state_dict vào một mạng đã dựng sẵn (dùng cho init từ checkpoint khác và resume)."""
    sd2, rep = adapt_state_dict(sd, net, renames)
    try:
        net.load_state_dict(sd2)
    except RuntimeError as e:
        raise RuntimeError(f"không nạp được checkpoint vào {type(net).__name__}. Mạng bản 1 và mạng phẳng chỉ "
                           f"nạp được với đúng file dữ liệu lúc huấn luyện.\n{e}") from e
    return rep


def checkpoint_config(ck: dict) -> dict:
    """Cấu hình của checkpoint, bổ sung mặc định. Checkpoint cũ không ghi net.kind là mạng bản 1."""
    cfg = ck["config"]
    net_cfg = dict(cfg.get("net", {}))
    net_cfg.setdefault("kind", "entity")
    cfg = deep_merge(DEFAULTS, cfg)
    cfg["net"] = deep_merge(DEFAULTS["net"], net_cfg)
    return cfg


def load_compatible(ck, cat, device="cpu", renames: dict | None = None, combat_version: str | None = None):
    """ck: đường dẫn hoặc dict checkpoint (của train.py hoặc bc.py). Trả về (net, cfg, report)."""
    from .nets import build_net
    if isinstance(ck, str):
        ck = torch.load(ck, map_location=device, weights_only=False)
    cfg = checkpoint_config(ck)
    net = build_net(cat, cfg["net"], cfg["num_players"])
    rep = load_into(net, ck["net"], renames)
    trained = cfg.get("versions", {})
    if combat_version and trained.get("combat") and trained["combat"] != combat_version:
        rep.warnings.append(f"module giao tranh khác lúc huấn luyện: {trained['combat']} -> {combat_version}")
    if trained.get("catalog") and trained["catalog"] != cat.version and not any("file dữ liệu" in w for w in rep.warnings):
        rep.warnings.append(f"file dữ liệu khác lúc huấn luyện: {trained['catalog']} -> {cat.version}")
    net.to(device)
    return net, cfg, rep
