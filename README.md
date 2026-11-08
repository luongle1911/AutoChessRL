# Auto Chess RL — giai đoạn 1: lõi vĩ mô + giao diện chơi thử

Code theo tài liệu *Thiết kế Auto Chess RL — Core & UI*, bước 1–5 của thứ tự triển khai
(dữ liệu → lõi → giao tranh formula-v1 → lớp chơi thử dòng lệnh), cùng tối ưu lõi, bốn bot luật, giải đấu Elo,
và phần học củng cố (`autochess/rl/`, theo tài liệu *Thiết kế Auto Chess RL — Agent vĩ mô*, bước 1–5).

## Cài đặt

Python ≥ 3.10, NumPy. `pytest` chỉ cần để chạy kiểm thử.

```bash
pip install numpy pytest
```

Mọi lệnh chạy từ thư mục gốc `AutoChessRL/`.

## Chơi thử

```bash
python -m autochess.play.cli play                     # bạn ghế 0, bot econ ghế 1
python -m autochess.play.cli play -p 4 --bot econ,random --seed 7
python -m autochess.play.cli play --no-matchup        # ẩn xác suất thắng (như tắt obs.matchup_row)
python -m autochess.play.cli watch -p 4               # toàn bot, xem từng bước
python -m autochess.play.cli replay logs/game_7_....json
```

Lệnh trong ván (mỗi lệnh = đúng một số hiệu hành động):

| Lệnh | Ý nghĩa | Số hiệu |
|---|---|---|
| `b i` | mua ô shop i (0–4) | i |
| `s j` | bán quân ở ô j (0–8 chờ, 9–17 sân) | 5 + j |
| `m j` | chuyển quân ở ô j lên/xuống sân | 23 + j |
| `r` | roll | 41 |
| `x` | mua XP | 42 |
| `e` | kết thúc lượt | 43 |
| `a n` | gửi thẳng số hiệu n (gỡ lỗi) | n |

Lệnh gỡ lỗi (không vào log): `?` (hành động hợp lệ), `help`, `undo`, `god`, `raw`, `check`, `save [file]`, `quit`.
Lệnh bị từ chối sẽ in lý do lấy từ mặt nạ thành phần. Hết ván hoặc thoát, log tự ghi vào `logs/`.

- watch: `Enter` 1 bước · `r` hết vòng · `a` hết ván · `v N` xem ghế N · `g` god · `check` · `save` · `q`
- replay: `n`/`p` vòng sau/trước · `j N` tới vòng N · `s` 1 bước (in cả hành động) · `v N` · `g` · `check` · `q`

Màu ANSI bật khi chạy trong terminal; tắt bằng `--no-color` hoặc biến môi trường `NO_COLOR`.
`--no-debug` tắt kiểm tra bất biến sau mỗi bước.

## Kiểm thử và đo tốc độ

```bash
python3 -m pytest autochess/tests -q                     # ~1 phút
FUZZ_GAMES=10000 python3 -m pytest autochess/tests/test_fuzz.py -q   # đủ 10.000 ván (chậm, chạy trước commit lớn)
python3 -m autochess.bench                               # bước-ván/giây với B = 1, 64, 1024 (bot ngẫu nhiên)
python3 -m autochess.bench --players 8 --profile
python3 -m autochess.bench --bots tempo,hyper,roll7,fast9 --players 8   # bot luật, sát thực tế hơn
```

`test_golden.py` so mã băm trạng thái của 100 ván (P = 2 và P = 8) với `golden.json`, tạo từ bản lõi
trước khi tối ưu. **Mọi lần tối ưu lõi phải giữ test này xanh**: kết quả mô phỏng không được đổi.
Chỉ tạo lại đáp án mẫu khi CỐ Ý đổi luật: `GOLDEN_REGEN=1 python3 -m pytest autochess/tests/test_golden.py`.

## Bot luật và giải đấu

Bốn bot trong `bots/scripted.py` dùng chung một lớp `ScriptedBot`, chỉ khác bộ tham số (`PRESETS`).
Bot đọc thẳng mảng trạng thái, vector hóa trên B ván và nhiều ghế, không có ngẫu nhiên.

| Bot | Đỉnh | Lối chơi |
|---|---|---|
| `tempo` | vòng 1-6 | không giữ vàng, không roll, dồn vàng lên cấp (cấp 6 ở vòng 6), mua quân mạnh nhất |
| `hyper` | vòng 7-11 | roll hết vàng từ vòng 3 ở cấp 4-6, săn 2 tướng 1-2 tiền lên 3 sao |
| `roll7` | vòng 12-16 | giữ 50 vàng lấy lãi, lên cấp 7 ở vòng 12 rồi roll xuống săn tướng 2-3 tiền |
| `fast9` | vòng 17+ | giữ lãi, lên cấp 8 ở vòng 16, cấp 9 ở vòng 22, săn tướng 4-5 tiền |
| `bank` | cả ván | giữ lãi tới 50, lên cấp đều, săn tướng 2-4 tiền ghép sao, máu dưới 50 thì tiêu hết |
| `tank` | vòng 13+ | "bán máu": đánh 2 vòng đầu, từ vòng 3 bỏ trống sân, thua liên tục để ăn thưởng chuỗi thua + lãi, vàng dư mua XP; vòng 13 lên cấp 8, roll hết dựng đội tướng 3-5 tiền |

```bash
python3 -m autochess.tournament            # ~2 phút: đường cong sức mạnh, thắng theo giai đoạn, Elo
python3 -m autochess.tournament --quick    # nhanh, ít ván hơn
python3 -m autochess.play.cli watch -p 4 --bot tempo,hyper,roll7,fast9
python3 -m autochess.play.cli play --bot hyper
```

Ba lớp kiểm thử: `test_rules.py` (dựng trạng thái tay từng luật), `test_fuzz.py` (bot ngẫu nhiên +
bất biến sau mỗi bước, tất định, nhất quán theo lô), `test_play.py` (lệnh, ẩn thông tin, undo, replay).

## Dùng lõi trực tiếp (cho huấn luyện)

```python
import numpy as np
from autochess.core import load_catalog, FormulaCombat, reset, step, legal_actions, rng

cat = load_catalog()
combat = FormulaCombat(catalog=cat)
state = reset(cat, rng.make_seeds(0, 1024), num_players=2)   # B = 1024 ván
mask = legal_actions(cat, state)                               # [B, P, 44]
actions = np.where(mask, np.random.rand(*mask.shape), -1).argmax(-1)
state, info = step(cat, combat, state, actions)               # ghi đè state tại chỗ
# info["reset"], info["final_placement"]: ván vừa xong ở bước trước và đã tự khởi tạo lại
```

## Học củng cố (agent vĩ mô)

Cần thêm PyTorch và PyYAML; TensorBoard là tùy chọn (không có thì chỉ ghi `metrics.jsonl`).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop: bản CPU, nhẹ hơn
pip install pyyaml tensorboard
python3 -m pytest autochess/tests/test_rl_core.py autochess/tests/test_rl_torch.py \
    autochess/tests/test_obs_v2.py autochess/tests/test_net_v2.py -q
```

Huấn luyện (mỗi lần chạy tạo `runs/<tên>_<thời gian>/`: cấu hình, `metrics.jsonl`, `eval.jsonl`, checkpoint):

```bash
# bước 3: kiểm tra đường ống, agent phải thắng bot ngẫu nhiên > 95% (mốc 1)
python3 -m autochess.rl.train --config configs/p2_random.yaml configs/laptop.yaml
# cùng việc đó với mạng phẳng đối chứng
python3 -m autochess.rl.train --config configs/p2_random.yaml configs/laptop.yaml --set net.kind=flat name=p2_random_flat
# giai đoạn 1: bốn bot luật, ưu tiên bot khó (mục tiêu mốc 3: > 60% trước cả bốn)
python3 -m autochess.rl.train --config configs/p2_bots.yaml configs/laptop.yaml
python3 -m autochess.rl.train --resume runs/p2_bots_...           # chạy tiếp sau khi dừng
# nếu agent không thắng nổi bot nào (thưởng cuối ván luôn -1): học bắt chước bot trước (~15 phút trên laptop)
python3 -m autochess.rl.bc --config configs/p2_bots.yaml configs/laptop.yaml
python3 -m autochess.rl.train --config configs/p2_bots.yaml configs/laptop.yaml \
    --set init=runs/bc_.../checkpoints/latest.pt ppo.value_warmup=50 ppo.lr_warmup=50
tensorboard --logdir runs                                          # xem biểu đồ (nếu đã cài)
python3 -m autochess.rl.evaluate --ckpt runs/.../checkpoints/latest.pt --games 1000
python3 -m autochess.rl.evaluate --ckpt <mạng A> --vs <mạng B> --games 1000    # hai mạng đấu trực tiếp
python3 -m autochess.rl.evaluate --ckpt <mạng bản 2> --catalog autochess/data/catalog_v1b.json   # thử bản cập nhật
python3 -m autochess.play.cli play --bot agent:runs/.../checkpoints/latest.pt   # tự đấu với agent
```

**Giai đoạn 2: tự đấu có kho phiên bản** (`configs/p2_selfplay.yaml`): 50% ván agent tự đấu, 35% đấu bản
cũ trong kho (bản neo + bản chụp mỗi 25 lần cập nhật, tối đa 10), 15% đấu bot luật (thêm bot `bank`: tích vàng
rồi xả). Bật sẵn chống thao tác thừa (`reward.churn_cost`, `ppo.ent_coef_final`). Đánh giá thêm với các bản neo
cố định (`eval.anchors`) và tính Elo chung, vì tự đấu thì tỷ lệ thắng luôn quanh 50%.

```bash
python3 -m autochess.rl.train --config configs/p2_selfplay.yaml configs/laptop.yaml
python3 -m autochess.rl.train --resume runs/p2_selfplay_...      # chạy tiếp (kho bản chụp được dựng lại)
```

**Luật bản 2** (`autochess/data/catalog_v2.json`): 3 sao tướng 1-3 tiền mạnh hơn, 1 sao tướng 4-5 tiền yếu hơn
nhưng có thêm tộc, thưởng tộc ×1,6, khắc chế ×1,5, giao tranh ít may rủi hơn (nhiệt độ 0,2).
**Luật bản 4** (`catalog_v4.json`, sinh bằng `python3 -m autochess.data.make_catalog_v4`): 59 tướng thần thoại
(13/13/13/12/8 theo giá), kho chung như ĐTCL (30/25/18/10/9), mỗi tướng 5 chỉ số (máu, sát thương vật lý,
sát thương phép, giáp, kháng phép; mỗi sao máu và sát thương x1,8), 24 tộc hệ (5 hệ vai trò, 11 tộc theo thần
thoại, 8 tộc riêng của tướng 5 tiền; mốc linh hoạt; hiệu ứng cộng chỉ số cho tướng mang tộc hoặc cả đội). Giao
tranh `ttk-v1`: đội hạ đối thủ nhanh hơn (máu chia sát thương sau giáp / kháng phép) thì nhiều khả năng thắng hơn,
cộng khắc chế 5x5 giữa các hệ. Kinh tế, máu mất, 50 hành động mỗi vòng như bản 3.2.

**Huấn luyện trên luật bản 4** (thẳng sảnh 8 người, không qua 1 đấu 1, không nạp agent cũ). Mạng `entity2` tự
chuyển sang quan sát bản 4 khi file luật có chỉ số: thêm 5 chỉ số cho từng quân (trên sân là chỉ số cuối sau sao
và tộc hệ), chỉ số đội và tỷ lệ hệ của mình / từng đối thủ, mức tăng độ mạnh đội nếu mua từng quân trong shop;
thẻ tộc hệ đọc bảng mốc + hiệu ứng theo chỉ số. Attention giữa các thực thể là tùy chọn (`net.mixer: attn`, đặt
giống nhau ở cả hai bước).

```bash
# bước 1 (~1 giờ trên laptop): học bắt chước tempo / bank / tank trong sảnh 8 người, chép kết quả ra runs/bc_v4.pt
nohup python3 -u -m autochess.rl.bc --config configs/p4_bc.yaml > bc_v4.log 2>&1 &
# bước 2 (~9-10 giờ cho 12 triệu quyết định): PPO trong sảnh 8 người (25% sảnh 4 người), đối thủ = bot luật +
# bản chụp của agent + bản học bắt chước, 40% tự đấu; đánh giá 3 sảnh A (bot) / B (bản chụp mới nhất) / C (bc + bot)
nohup python3 -u -m autochess.rl.train --config configs/p4_league.yaml > p4.log 2>&1 &
```

**Giao diện web để chơi thử** (không cần cài thêm gì):

```bash
python3 -m autochess.play.web --catalog autochess/data/catalog_v4.json -p 8 --seed 1 \
    --bot tempo,bank,tank,fast9,roll7,hyper,bank
# mở http://localhost:8000 ; phím: 1-5 mua, D roll, F mua kinh nghiệm, Space kết thúc lượt, Z hoàn tác,
# bấm quân rồi W (lên sân / xuống chờ), E (bán); bấm tộc hệ bên trái / đối thủ bên phải để xem chi tiết
```

**Luật bản 3** (`catalog_v3.json`, dùng cho giai đoạn 3 từ 06/11): 3 sao tướng 1 / 2 tiền 72 / 98, kho chung
tướng 1-2 tiền lớn hơn, nhiệt độ 0,17, thua mất ít máu hơn và tăng dần chậm hơn (cả phần cố định lẫn phần
theo số quân: `unit_damage_scale`), nên ván 8 người dài ~28 vòng thay vì ~23,5 (đủ thời gian xả vàng, lên cấp 8-9).
Bản 3.2: tối đa 50 hành động mỗi vòng (trước 30). Hàng chờ đối thủ hiện công khai
trong giao diện chơi thử (như ĐTCL).
`python3 -m autochess.rl.patchtest --ckpt ... --catalog ...` đo agent trên một file dữ liệu (thắng bot, đấu trực
tiếp, phần sức mạnh từ tộc, tỷ lệ mua từng tướng).

**Giai đoạn 3: sảnh 8 người** (`configs/p3_lobby.yaml`, luật bản 2): 75% ván 8 người, 25% ván 4 người (đệm
thành "đối thủ đã bị loại" để học chung). Một nửa số ván toàn agent, nửa còn lại agent ngồi 1-2 ghế, các ghế
khác bốc từ kho. Agent thấy hàng chờ đối thủ (`net.opp_bench`, quan sát bản 3; nạp mạng bản 2 thì cột mới = 0).
`env.skip_waiting` bỏ các bước chờ người khác xong lượt khỏi lần học (nhanh gấp ~2 ở 8 người). Đánh giá trong
các sảnh cố định `eval.lobbies` (A: bot luật, B: 7 bản sao điểm xuất phát, C: bản cũ + bot) với Elo theo cặp,
cộng 1 đấu 1 với bản neo để biết agent không quên. Mốc 5: top 4 trên 60% ở sảnh B.

```bash
python3 -u -m autochess.rl.train --config configs/p3_lobby.yaml configs/laptop.yaml --set env.num_games=128 eval.games=200
python3 -m autochess.rl.evaluate --ckpt runs/p3_lobby_.../checkpoints/latest.pt --catalog autochess/data/catalog_v2.json \
    --lobbies configs/p3_lobby.yaml --games 256 --B 64                 # đánh giá sảnh riêng
# PC: mạng học trên GPU, thu thập bằng 12 tiến trình (rl/vec.py); lần đầu chạy kiểm thử GPU:
python3 -m pytest -q autochess/tests/test_stage3.py -k gpu
python3 -u -m autochess.rl.train --config configs/p3_lobby.yaml configs/pc.yaml
```

`--set net.critic_info=pairing` (hoặc `pairing+dice`): critic biết thêm đối thủ thật của vòng (và xúc xắc của
trận). Chính sách không thấy thông tin này (có kiểm thử); mặc định tắt, để so sánh khi có PC.

**Mạng bản 2 (mặc định, `net.kind=entity2`).** Không tham số nào phụ thuộc số tướng hay số tộc: thẻ
tướng/tộc tính từ chỉ số trong file dữ liệu, cộng một "phần riêng" nhỏ lưu theo tên. Vì vậy checkpoint bản 2
nạp được sau khi thêm/bớt/chỉnh tướng hay tộc (`rl/compat.py` ghép theo tên và in báo cáo giữ/mới/bỏ).
Mạng bản 1 (`net.kind=entity`) và checkpoint cũ của nó vẫn chạy như trước, nhưng chỉ với đúng file dữ liệu cũ.
`--set net.mixer=attn` thêm 2 lớp attention giữa các thực thể (chậm hơn ~5 lần trên CPU, để dành cho PC).
`--set catalog=<file>` đổi file dữ liệu; `autochess/data/catalog_v1b.json` là một bản cập nhật giả để thử.

Mỗi dòng in ra khi huấn luyện: số quyết định, tốc độ, các chỉ số PPO (KL ~0,01–0,02, tỷ lệ kẹp ~0,1–0,2,
entropy giảm từ từ, EV của critic tăng dần) và tỷ lệ thắng trong các ván vừa chơi. Mỗi `eval.every` lần
cập nhật có một lần đánh giá đầy đủ (tỷ lệ thắng theo bot và theo giai đoạn, Elo, hành vi, các mốc).

| File | Việc |
|---|---|
| `rl/observe.py` | quan sát nhiều mảng (bản 1, 2, 3, `version=`); kho chung tính theo những gì nhìn thấy được; `critic_info` |
| `rl/reward.py` | thưởng thứ hạng, thưởng phụ thế năng (tổng bằng 0 mỗi tập), γ/λ chỉ trôi khi có giao tranh |
| `rl/gae.py` | GAE theo vòng, bước của người đã bị loại/đang chờ reset là trong suốt |
| `rl/nets.py` | `EntityNetV2` (~353 nghìn tham số, ~361 nghìn khi thấy hàng chờ đối thủ, ~429 nghìn với attention), `EntityNetV1` (~309 nghìn), `FlatNet` |
| `rl/compat.py` | `load_compatible`: nạp checkpoint khi file dữ liệu đã đổi, ghép phần riêng theo tên |
| `rl/policy.py`, `rl/agent.py` | bọc mạng thành chính sách; agent ngồi ghế như bot |
| `rl/opponents.py` | bot đối thủ, kho đối thủ, ưu tiên đối thủ khó (1 − tỷ lệ thắng)², chia nhóm bot / bản cũ |
| `rl/league.py` | giai đoạn 2: bản chụp đóng băng của agent làm đối thủ (chụp, thay, chạy tiếp) |
| `rl/rollout.py` | thu thập một tiến trình; nhiều ghế agent mỗi ván; `Lobbies` gộp ván 8 và 4 người; `skip_waiting` |
| `rl/vec.py` | thu thập nhiều tiến trình (`workers=`), đồng bộ kho bản chụp sang tiến trình con |
| `rl/ppo.py`, `rl/train.py`, `rl/config.py` | PPO kiểu CleanRL có mặt nạ, vòng huấn luyện, cấu hình |
| `rl/evaluate.py` | bộ đánh giá, Elo cùng bảng bot; `evaluate_lobbies`: sảnh nhiều người, Elo theo cặp |
| `play/web.py`, `play/web_index.html` | giao diện web chơi thử (máy chủ thư viện chuẩn + một trang HTML) |
| `data/make_catalog_v4.py` | sinh luật bản 4 (bảng tướng, chỉ số, tộc hệ) |
| `rl/patchtest.py` | đo một agent trên một file dữ liệu (bản cập nhật luật) |
| `rl/botdata.py`, `rl/bc.py` | học bắt chước bot luật để khởi động agent; `ppo.value_warmup` (chỉ critic học, chính sách đứng yên) rồi `ppo.lr_warmup` |
| `configs/` | `p2_random`, `p2_bots`, `p2_selfplay`, `p3_lobby`, `laptop`, `pc` (gộp theo thứ tự, `--set khóa=giá trị` để ghi đè) |

## Cấu trúc

```
autochess/
  data/catalog_v1.json   20 tướng, 6 tộc, mọi hằng số luật
  core/                  chỉ NumPy, không import gì từ play/
    catalog.py  rng.py  state.py  actions.py  step.py  combat.py  pairing.py  invariants.py
  play/                  B = 1
    adapter.py  seats.py  runner.py  replay.py  cli.py
  bots/                  random_bot.py, econ_bot.py (đọc view), scripted.py (4 bot vector hóa)
  tests/                 + test_golden.py (đáp án mẫu), test_bots.py
  bench.py               tốc độ lõi
  tournament.py          giải đấu bot, Elo
  rl/                    học củng cố (xem mục trên)
configs/                 cấu hình huấn luyện
```

## Chỗ khác hoặc cụ thể hóa so với tài liệu thiết kế

1. `step` trả về `(state, info)` thay vì `(state, events)`; `info["events"]` là danh sách sự kiện
   (chỉ có khi B = 1 hoặc `debug=True`), kèm `reset`, `final_placement`, `combat`, `opp`, `ghost`, `win_prob`.
2. Điểm khắc chế dùng mức kích hoạt tộc chia về 0..1 (mốc đạt được / 3) để cùng thang với log tỷ lệ sức mạnh.
3. Kho chung cho P = 3..7 được nội suy tuyến tính giữa cấu hình 2 và 8 người.
4. Ván đã kết thúc được khởi tạo lại ở lần gọi `step` kế tiếp, seed mới = băm(seed cũ); hành động gửi cho ván đó ở bước này bị bỏ qua.
5. Roll miễn phí đầu vòng dùng "chỉ số bước" 1000 trong khóa ngẫu nhiên để không trùng các lần roll thường.
6. Ở chế độ không gỡ lỗi, hành động bị che được thay bằng kết thúc lượt; chế độ gỡ lỗi báo lỗi ngay.
7. `undo` lùi về trước lệnh gần nhất bạn gõ; hành động của bot ở cùng các bước đó cũng bị bỏ và bot có thể chọn khác.
8. Chưa dùng `rich`; giao diện là văn bản thuần với màu ANSI tùy chọn.
9. `observe.py` chưa có, nên `PolicySeat` đã có khung nhưng chưa nhận được quan sát.
10. `step(..., mask=m)` nhận mặt nạ đã tính sẵn (trên chính trạng thái đó) để khỏi tính hai lần.
11. Module giao tranh chỉ được gọi cho các ván đang giao tranh, qua một lát cắt chỉ-đọc
    (`units`, `round`, `alive`, `hp`, `level`); module mới phải chỉ dùng các mảng này.
