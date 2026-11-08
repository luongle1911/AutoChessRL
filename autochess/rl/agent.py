"""Agent ngồi vào ghế như một bot (không cần PyTorch: chính sách được truyền vào từ ngoài)."""
from __future__ import annotations

import numpy as np

from autochess.core.actions import END_TURN, ROLL, BUY_XP, BUY0, SELL0, MOVE0


class AgentPlayer:
    """Agent ngồi vào ghế như một bot: dùng được trong tournament.play, bộ đánh giá và lớp chơi thử.
    policy: đối tượng có act(obs, rng, greedy) -> (actions, logp, value), ví dụ rl.policy.Policy.

    act(state, mask[B, P, 44], seats) -> actions[B, len(seats)]
    act_one(state, mask_b[44], b, seat) -> int   (cho ArrayBotSeat trong giao diện dòng lệnh)
    Có ghi thống kê hành vi: số roll mỗi vòng, vàng khi kết thúc lượt, cấp theo vòng; từ vòng LATE_ROUND trở đi
    (lúc kết thúc lượt): số quân 3 sao / 2 sao trên sân, số tộc hệ đang kích hoạt (không tính tộc riêng một tướng),
    và "lối chơi chính" = tộc hệ (không riêng) có mức kích hoạt cao nhất so với số mốc của nó.
    """
    LATE_ROUND = 20

    def __init__(self, policy, cat, combat, matchup_row: bool = True, seed: int = 0,
                 greedy: bool = False, name: str = "agent"):
        self.policy, self.cat, self.combat = policy, cat, combat
        self.matchup_row, self.greedy, self.name = matchup_row, greedy, name
        self.rng = np.random.default_rng(seed)
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(rolls=0, xp_buys=0, buys=0, sells=0, moves=0, decisions=0, end_turns=0,
                          gold_at_end=0.0, level_by_round={}, actions_hit_cap=0,
                          late_turns=0, late_3star=0, late_2star=0, late_traits=0, main_trait={})

    def act(self, state, mask, seats):
        from .observe import observe, flatten_batch
        seats = np.atleast_1d(np.asarray(seats))
        B = state.B
        seat_arr = np.tile(seats, (B, 1))
        obs = observe(self.cat, state, self.combat, self.matchup_row, seats=seat_arr, mask=mask,
                      version=getattr(self.policy, "obs_version", 1))
        a, _, _ = self.policy.act(flatten_batch(obs), self.rng, greedy=self.greedy)
        a = a.reshape(B, len(seats))
        self._record(state, mask, seat_arr, a)
        return a

    def act_one(self, state, mask_b, b, seat):
        sb = state.rows(b) if state.B > 1 else state
        full = np.zeros((1, sb.P, mask_b.shape[-1]), bool)
        full[0, seat] = mask_b
        full[0, :, END_TURN] = True
        return int(self.act(sb, full, [seat])[0, 0])

    def _record(self, state, mask, seats, a):
        b = np.arange(state.B)[:, None]
        live = state.alive[b, seats] & ~state.done[:, None] & ~state.ended_turn[b, seats]
        dec = live & (mask[b, seats].sum(-1) > 1)
        st = self.stats
        st["decisions"] += int(dec.sum())
        st["rolls"] += int((dec & (a == ROLL)).sum())
        st["xp_buys"] += int((dec & (a == BUY_XP)).sum())
        st["buys"] += int((dec & (a >= BUY0) & (a < SELL0)).sum())
        st["sells"] += int((dec & (a >= SELL0) & (a < MOVE0)).sum())
        st["moves"] += int((dec & (a >= MOVE0) & (a < ROLL)).sum())
        end = live & (a == END_TURN)
        st["end_turns"] += int(end.sum())
        st["gold_at_end"] += float(state.gold[b, seats][end].sum())
        st["actions_hit_cap"] += int((live & (state.actions_this_round[b, seats]
                                               >= self.cat.max_actions_per_round - 1)).sum())
        late = end & (state.round[:, None] >= self.LATE_ROUND)
        if late.any():
            self._record_board(state, seats, late)
        for r, lv in zip(state.round[np.nonzero(end)[0]], state.level[b, seats][end]):
            d = st["level_by_round"].setdefault(int(r), [0, 0])
            d[0] += int(lv)
            d[1] += 1

    def _record_board(self, state, seats, sel):
        from autochess.core.catalog import BENCH_SIZE
        from autochess.core.combat import _board_features
        cat, st = self.cat, self.stats
        bi, si = np.nonzero(sel)
        board = state.units[bi, seats[bi, si], BENCH_SIZE:].astype(np.int64)            # [M,9]
        occ = board >= 0
        star = np.where(occ, board % 3, -1)
        f = _board_features(cat, board)
        kinds = [t.get("kind", "trait") for t in sorted(cat.raw["traits"], key=lambda t: t["id"])]
        normal = np.array([k != "unique" for k in kinds])
        lvl = f["trait_level"]
        n_lv = np.asarray(cat.trait_levels if cat.trait_levels is not None
                          else np.full(cat.K, cat.trait_breakpoints.shape[1]))
        st["late_turns"] += len(bi)
        st["late_3star"] += int((star == 2).sum())
        st["late_2star"] += int((star == 1).sum())
        st["late_traits"] += int(((lvl > 0) & normal).sum())
        score = np.where(normal, lvl / n_lv + f["trait_count"] * 1e-3, -1.0)
        main = score.argmax(-1)
        for k, ok in zip(main, lvl[np.arange(len(main)), main] > 0):
            name = cat.trait_names[k] if ok else "(không)"
            st["main_trait"][name] = st["main_trait"].get(name, 0) + 1

    def summary(self) -> dict:
        st = self.stats
        rounds = max(st["end_turns"], 1)
        lv = {r: round(s / n, 2) for r, (s, n) in sorted(st["level_by_round"].items()) if n >= 10}
        return dict(rolls_per_round=st["rolls"] / rounds, xp_per_round=st["xp_buys"] / rounds,
                    buys_per_round=st["buys"] / rounds, sells_per_round=st["sells"] / rounds,
                    moves_per_round=st["moves"] / rounds,
                    gold_at_end_turn=st["gold_at_end"] / rounds,
                    decisions_per_round=st["decisions"] / rounds,
                    cap_hits_per_round=st["actions_hit_cap"] / rounds,
                    level_at_round={r: lv[r] for r in (5, 10, 15, 20) if r in lv},
                    **self.late_summary())

    def late_summary(self) -> dict:
        st = self.stats
        n = st["late_turns"]
        if not n:
            return {}
        mt = sorted(st["main_trait"].items(), key=lambda x: -x[1])
        share = np.array([c for _, c in mt], np.float64) / n
        ent = float(-(share * np.log(share)).sum())
        return dict(late_3star=st["late_3star"] / n, late_2star=st["late_2star"] / n,
                    late_active_traits=st["late_traits"] / n,
                    main_traits={k: round(c / n, 3) for k, c in mt[:6]},
                    comp_diversity=round(float(np.exp(ent)), 2))
