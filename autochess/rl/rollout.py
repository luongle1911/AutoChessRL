"""Một bộ thu thập (worker): giữ B ván song song, mỗi lần collect() chạy T bước và trả dữ liệu cho PPO.

Mỗi ván có:
    owner[b, p]       -1 = ghế của agent (chính sách đang học), k >= 0 = đối thủ thứ k trong OpponentPool
    agent_seats[b, s] các ghế của agent (S ô; -1 = ô không dùng ở ván này)
Khi một ván kết thúc, kết quả được ghi vào OpponentPool (cho ưu tiên đối thủ khó) và ván kế tiếp được
gán lại ghế/đối thủ. Ghế agent luân phiên giữa các ván để không lệch ghế.

Mỗi ván: với xác suất selfplay_prob mọi ghế là agent (ván tự đấu); còn lại agent ngồi agent_seats = (ít
nhất, nhiều nhất) ghế, các ghế khác bốc từ kho. Ván nhiều người: agent xếp trên đối thủ nào là thắng đối
thủ đó (kho ghi theo từng cặp). Mỗi bước, mọi ghế của cùng một đối thủ (ở mọi ván) được gộp vào MỘT lần
gọi đối thủ đó.

Lobbies gộp nhiều Rollout có số người khác nhau (vd 8 người và 4 người): xem class Lobbies.

Chính sách (policy) chỉ cần hai hàm, nhận quan sát đã trải phẳng [M, ...]:
    act(obs, rng)  -> actions[M], logp[M], value[M]   (bốc theo phân phối)
    value(obs)     -> value[M]
"""
from __future__ import annotations

import time

import numpy as np

from autochess.core import reset, step, legal_actions, rng as crng
from autochess.core.actions import END_TURN, NUM_ACTIONS, SELL0, ROLL
from .observe import observe, flatten_batch
from .reward import RewardTracker


class RandomPolicy:
    """Chính sách đều trên các hành động hợp lệ (dùng cho kiểm thử, không cần PyTorch)."""
    obs_version = 1

    def act(self, obs, rng, greedy=False):
        m = obs["mask"]
        g = np.where(m, rng.random(m.shape), -1.0)
        a = g.argmax(-1)
        n = m.sum(-1)
        return a, -np.log(np.maximum(n, 1)).astype(np.float32), np.zeros(len(a), np.float32)

    def value(self, obs):
        return np.zeros(len(obs["mask"]), np.float32)


class Rollout:
    def __init__(self, cat, combat, pool, num_games: int, num_players: int = 2, steps: int = 128,
                 seed: int = 0, gamma_round: float = 0.995, lam: float = 0.95, beta: float = 0.1,
                 matchup_row: bool = True, selfplay_prob: float = 0.0, churn_cost: float = 0.0,
                 agent_seats=(1, 1), skip_waiting: bool = False):
        self.cat, self.combat, self.pool = cat, combat, pool
        # skip_waiting: chỉ chạy mạng ở bước có quyết định và bước chờ ĐẦU TIÊN sau đó (vừa kết thúc lượt);
        # các bước chờ tiếp theo (đợi người khác xong lượt, rất nhiều ở ván 8 người) dùng lại giá trị critic
        # đã tính và không đưa vào lô PPO (buf["key"] = False). GAE vẫn đi qua mọi bước nên không mất thưởng.
        self.skip_waiting = bool(skip_waiting)
        self.B, self.P, self.T = num_games, num_players, steps
        lo, hi = (int(agent_seats), int(agent_seats)) if np.isscalar(agent_seats) else map(int, agent_seats)
        self.seat_range = (max(1, min(lo, num_players - 1)), max(1, min(hi, num_players - 1)))
        self.S = num_players if selfplay_prob > 0 else self.seat_range[1]
        self.selfplay_prob = selfplay_prob
        self.matchup_row = matchup_row
        # thưởng âm rất nhỏ cho mỗi lần bán / chuyển quân: hai thao tác này gần như miễn phí trong luật nên
        # agent có thể chèn chúng vô nghĩa (đẩy bởi thưởng entropy) tới khi chạm trần hành động mỗi vòng
        self.churn_cost = float(churn_cost)
        self.rng = np.random.default_rng(seed)
        self.reward = RewardTracker(cat, num_players, gamma_round, lam, beta)
        self.state = reset(cat, crng.make_seeds(seed, num_games), num_players)
        self.owner = np.full((self.B, self.P), -1, np.int64)
        self.agent_seats = np.full((self.B, self.S), -1, np.int64)
        self.games_started = np.zeros(self.B, np.int64)
        self._assign(np.arange(self.B))
        self.buf = None
        self.finished: list[dict] = []   # kết quả các ván đã xong kể từ lần collect trước
        self.total_steps = 0

    # ------------------------------------------------------------------ gán ghế
    def _assign(self, rows: np.ndarray) -> None:
        P, S = self.P, self.S
        for b in rows:
            self.owner[b] = -1
            self.agent_seats[b] = -1
            selfplay = self.selfplay_prob > 0 and self.rng.random() < self.selfplay_prob
            if selfplay:
                self.agent_seats[b, :P] = np.arange(P)
            else:
                lo, hi = self.seat_range
                n = int(self.rng.integers(lo, hi + 1))
                seat = int((b + self.games_started[b]) % P)
                rest = [p for p in range(P) if p != seat]
                extra = self.rng.choice(rest, n - 1, replace=False).tolist() if n > 1 else []
                mine = [seat] + extra
                self.agent_seats[b, :n] = mine
                others = [p for p in range(P) if p not in mine]
                self.owner[b, others] = self.pool.sample(len(others), self.rng)
            self.games_started[b] += 1

    # ------------------------------------------------------------------ thu thập
    def _alloc(self, obs):
        T, B, S = self.T, self.B, self.S
        self.buf = {"obs": {k: np.zeros((T, B, S) + v.shape[2:], v.dtype) for k, v in obs.items()}}
        for k, dt in (("actions", np.int64), ("logp", np.float32), ("values", np.float32),
                      ("rewards", np.float32), ("dones", bool), ("gammas", np.float32),
                      ("lambdas", np.float32), ("active", bool), ("decision", bool), ("key", bool)):
            self.buf[k] = np.zeros((T, B, S), dt)

    def _observe(self, mask, version=1, critic_info="none"):
        seats = np.where(self.agent_seats >= 0, self.agent_seats, 0)
        return observe(self.cat, self.state, self.combat, self.matchup_row, seats=seats, mask=mask,
                       version=version, critic_info=critic_info)

    def collect(self, policy) -> dict:
        B, P, S, T = self.B, self.P, self.S, self.T
        ver = getattr(policy, "obs_version", 1)
        ci = getattr(policy, "critic_info", "none")
        if self.buf is not None and getattr(self, "_buf_version", None) != (ver, ci):
            self.buf = None
        bidx = np.arange(B)[:, None]
        t0 = time.perf_counter()
        t_policy = 0.0
        if not hasattr(self, "_v_cache"):
            self._v_cache = np.zeros((B, S), np.float32)
            self._fresh = np.zeros((B, S), bool)       # _v_cache còn đúng cho trạng thái chờ hiện tại
        for t in range(T):
            s = self.state
            m = legal_actions(self.cat, s)
            obs = self._observe(m, ver, ci)
            if self.buf is None:
                self._alloc(obs)
                self._buf_version = (ver, ci)
            pre = self.reward.before(s, self.agent_seats)
            active = pre["active"]
            n_legal = m[bidx, pre["ss"]].sum(-1)
            decision = active & (n_legal > 1)

            tp = time.perf_counter()
            fobs = flatten_batch(obs)
            if self.skip_waiting:
                key = decision | (active & ~self._fresh)
                sel = np.flatnonzero(key.reshape(-1))
                a_flat = np.full(B * S, END_TURN, np.int64)
                logp = np.zeros(B * S, np.float32)
                val = self._v_cache.reshape(-1).copy()
                if sel.size:
                    a_s, l_s, v_s = policy.act({k: v[sel] for k, v in fobs.items()}, self.rng)
                    a_flat[sel], logp[sel], val[sel] = a_s, l_s, v_s
                self._v_cache = val.reshape(B, S)
                # sau bước có quyết định trạng thái còn đổi; sau một bước chờ thì giữ nguyên tới giao tranh
                self._fresh = active & ~decision
            else:
                key = active
                a_flat, logp, val = policy.act(fobs, self.rng)
            t_policy += time.perf_counter() - tp
            a_agent = a_flat.reshape(B, S)
            a_agent = np.where(active, a_agent, END_TURN)

            actions = np.full((B, P), END_TURN, np.int64)
            used = self.agent_seats >= 0
            rb, rs = np.nonzero(used)
            actions[rb, self.agent_seats[rb, rs]] = a_agent[rb, rs]
            # mỗi đối thủ một lần gọi cho mọi ghế của nó (một ván có thể xuất hiện hai lần nếu đối thủ đó
            # ngồi hai ghế: s.rows chép hàng, đối thủ chỉ đọc)
            live = ~s.done[:, None] & s.alive & ~s.ended_turn
            for k, opp in enumerate(self.pool.opponents):
                ob, op = np.nonzero((self.owner == k) & live)
                if ob.size:
                    actions[ob, op] = opp.act(s.rows(ob), m[ob], op)

            for k, v in obs.items():
                self.buf["obs"][k][t] = v
            self.buf["actions"][t] = a_agent
            self.buf["logp"][t] = logp.reshape(B, S)
            self.buf["values"][t] = val.reshape(B, S) * active
            self.buf["active"][t] = active
            self.buf["decision"][t] = decision
            self.buf["key"][t] = key

            self.state, info = step(self.cat, self.combat, s, actions, mask=m)
            r, done, g, l = self.reward.after(self.state, info, pre)
            if self.churn_cost:
                churn = decision & (a_agent >= SELL0) & (a_agent < ROLL)
                r = r - self.churn_cost * churn
            self.buf["rewards"][t] = r
            self.buf["dones"][t] = done
            self.buf["gammas"][t] = g
            self.buf["lambdas"][t] = l
            if self.skip_waiting:
                # giao tranh / hết ván: trạng thái chờ cũ không còn
                self._fresh &= ~(info["combat"] | self.state.done)[:, None]
            self._book_finished(info)

        # giá trị ở trạng thái cuối để ước lượng phần còn lại của các ván bị cắt giữa chừng
        m = legal_actions(self.cat, self.state)
        obs = self._observe(m, ver, ci)
        pre = self.reward.before(self.state, self.agent_seats)
        last_v = policy.value(flatten_batch(obs)).reshape(B, S) * pre["active"]
        self.total_steps += T
        sec = time.perf_counter() - t0
        finished, self.finished = self.finished, []
        return dict(buf=self.buf, last_values=last_v.astype(np.float32), finished=finished,
                    seconds=sec, policy_seconds=t_policy)

    # ------------------------------------------------------------------ kết quả ván
    def _book_finished(self, info) -> None:
        s = self.state
        ended = np.flatnonzero(s.done)
        if ended.size == 0:
            return
        # s.done vẫn True cho tới bước kế tiếp (khi lõi reset); chỉ ghi một lần
        if not hasattr(self, "_booked"):
            self._booked = np.zeros(self.B, bool)
        new = ended[~self._booked[ended]]
        self._booked[:] = s.done
        for b in new:
            place = s.placement[b]
            seats = [int(x) for x in self.agent_seats[b] if x >= 0]
            opp_seats = [p for p in range(self.P) if self.owner[b, p] >= 0]
            for a in seats:
                for o in opp_seats:
                    self.pool.update([self.owner[b, o]], [float(place[a] < place[o])])
            self.finished.append(dict(
                players=self.P, rounds=int(s.round[b]), agent_placement=[int(place[a]) for a in seats],
                opponents=[self.pool.names[self.owner[b, o]] for o in opp_seats],
                opp_placement=[int(place[o]) for o in opp_seats], selfplay=not opp_seats))
        self._assign(new)


class Lobbies:
    """Nhiều Rollout chạy lần lượt (vd ván 8 người và ván 4 người), cùng một chính sách và một kho đối thủ.
    collect() trả về danh sách kết quả của từng Rollout; train.py tính GAE riêng rồi đệm chiều đối thủ
    của ván ít người (observe.pad_opponents) để học chung một lần."""

    def __init__(self, rollouts: list):
        self.rollouts = list(rollouts)

    @property
    def total_steps(self):
        return sum(r.total_steps for r in self.rollouts)

    def collect(self, policy) -> list:
        return [r.collect(policy) for r in self.rollouts]
