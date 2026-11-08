"""Bất biến của trạng thái và băm trạng thái (để kiểm tra tất định)."""
from __future__ import annotations

import hashlib

import numpy as np

from .catalog import Catalog, BENCH_SIZE, MAX_LEVEL, STAR_COPIES
from .state import State, FIELD_NAMES


def state_hash(state: State, b: int | None = None) -> str:
    """SHA-256 của toàn bộ mảng (hoặc chỉ ván b)."""
    h = hashlib.sha256()
    for name in FIELD_NAMES:
        a = getattr(state, name)
        if b is not None:
            a = a[b]
        h.update(name.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def champion_totals(catalog: Catalog, state: State) -> np.ndarray:
    """[B, N]: kho + mọi shop + bản gốc trên tay mọi người."""
    B, N = state.pool.shape
    tot = state.pool.astype(np.int64).copy()
    sh = state.shop.reshape(B, -1)
    tot += ((sh[..., None] == np.arange(N)) & (sh[..., None] >= 0)).sum(1)
    u = state.units.reshape(B, -1)
    occ = u >= 0
    champ = np.where(occ, u // 3, -1)
    copies = np.where(occ, STAR_COPIES[np.where(occ, u % 3, 0)], 0)
    tot += ((champ[..., None] == np.arange(N)) * copies[..., None]).sum(1)
    return tot


def check_invariants(catalog: Catalog, state: State) -> list[str]:
    """Trả về danh sách vi phạm (rỗng = ổn). Vector hóa trên B, chỉ lặp để tạo thông báo."""
    s, errs = state, []
    B, P = s.B, s.P
    N = catalog.N

    def report(cond: np.ndarray, msg: str):
        bad = np.argwhere(cond)
        for idx in bad[:3]:
            errs.append(f"{msg} @ {tuple(int(x) for x in idx)}")
        if len(bad) > 3:
            errs.append(f"{msg}: còn {len(bad) - 3} chỗ khác")

    # bảo toàn tướng
    tot = champion_totals(catalog, s)
    report(tot != catalog.pool_size(P)[None, :], "bảo toàn tướng bị vi phạm (ván, tướng)")
    report(s.pool < 0, "kho âm (ván, tướng)")

    # miền giá trị
    report((s.units < -1) | (s.units >= 3 * N), "mã quân ngoài miền (ván, người, ô)")
    report((s.shop < -1) | (s.shop >= N), "mã shop ngoài miền (ván, người, ô)")
    report(s.gold < 0, "vàng âm (ván, người)")
    report(s.alive & ((s.hp <= 0) | (s.hp > catalog.start_hp)), "máu người còn sống ngoài (0, 100] (ván, người)")
    report((s.level < 1) | (s.level > MAX_LEVEL), "cấp ngoài 1..9 (ván, người)")
    need = catalog.xp_to_next[np.clip(s.level, 1, MAX_LEVEL) - 1]
    report((s.xp < 0) | ((s.level < MAX_LEVEL) & (s.xp >= need)), "XP sai (âm hoặc lẽ ra đã lên cấp) (ván, người)")
    report(s.actions_this_round > catalog.max_actions_per_round, "vượt giới hạn hành động (ván, người)")

    # sân & ghép sao
    board_n = (s.units[..., BENCH_SIZE:] >= 0).sum(-1)
    report(s.alive & (board_n > s.level), "số quân trên sân > cấp (ván, người)")
    u = s.units
    cnt = ((u[..., None] == np.arange(3 * N)) & (u[..., None] >= 0)).sum(-2)   # [B,P,3N]
    report(cnt[..., :] * (np.arange(3 * N) % 3 < 2) >= 3, "còn 3 quân cùng mã chưa ghép (ván, người, mã)")

    # người bị loại
    dead = ~s.alive
    report(dead & (s.units >= 0).any(-1), "người bị loại còn quân (ván, người)")
    report(dead & (s.shop >= 0).any(-1), "người bị loại còn shop (ván, người)")
    report(dead & (s.placement == 0), "người bị loại chưa có thứ hạng (ván, người)")
    report(~s.done[:, None] & s.alive & (s.placement != 0), "người còn sống có thứ hạng khi ván chưa xong (ván, người)")
    report(dead & ~s.ended_turn, "người bị loại chưa đánh dấu xong lượt (ván, người)")

    # ván đã xong: thứ hạng là hoán vị 1..P
    if s.done.any():
        pl = np.sort(s.placement[s.done].astype(np.int64), 1)
        bad = (pl != np.arange(1, P + 1)).any(1)
        rows = np.flatnonzero(s.done)[bad]
        for b in rows[:3]:
            errs.append(f"thứ hạng không phải hoán vị 1..P @ ván {b}: {s.placement[b].tolist()}")
    report((s.round < 1) | (s.round > catalog.max_rounds), "vòng ngoài 1..max (ván)")
    return [e for e in errs if e]
