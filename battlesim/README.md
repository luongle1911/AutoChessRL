# Mô phỏng giao tranh cho agent xếp quân

Đã xong M0–M3 theo kế hoạch: hình học lục giác, cấu hình, nấc 0 và nấc 1, trình xem lại trận, chí mạng ngẫu nhiên theo seed (cờ của nấc 3). Nấc 2 (năng lượng, kỹ năng, tộc/hệ) là M4.

## Cấu trúc

```
battlesim/
  layout.py     chỉ số cột của mảng trạng thái F (thực) và I (nguyên), mã sự kiện, mã kết quả
  hexgrid.py    bảng khoảng cách, hàng xóm, xoay 180° (tính một lần lúc import)
  config.py     đọc data/champions.json thành bảng numpy
  sim.py        kernel kiểu Numba (_init_one, _step_one, _run_all) + lớp Simulator
  jit.py        bọc Numba; tự lùi về Python thường nếu chưa cài
  viewer.py     trình xem lại: chạy một trận, xuất file HTML, mở bằng trình duyệt
  data/champions.json   6 tướng mẫu, 3 tộc/hệ mẫu, hệ số sao, dt, số bước tối đa
  tests/        hình học, luật tính tay, đối xứng, bất biến (Hypothesis), so khớp JIT, trình xem lại
```

## Cài thêm

```
pip install numba hypothesis
```
Không có Numba code vẫn chạy (chậm hơn); không có Hypothesis thì kiểm thử bất biến tự bỏ qua.

## Dùng

```python
from battlesim import load_config, Simulator, make_batch
cfg = load_config()
# mỗi tướng: (mã tướng, sao, ô nửa sân 0–27); hàng 0 = sau, hàng 3 = đầu
team_a = [(0, 2, 24), (2, 1, 3)]
team_b = [(1, 1, 24), (3, 1, 3)]
sim = Simulator(cfg, level=1, stochastic=False, log=True)
r = sim.run(make_batch([team_a], [team_b]))
r.result      # 0 / 1 = phe thắng, 2 = cả hai cùng thua
r.alive, r.star, r.dmg_dealt, r.dmg_taken, r.death_step, r.events
```

Chạy từng bước (kiểm thử, trình xem lại): `st = sim.init_state(batch)` rồi `sim.step(st, s)` với s = 1, 2, …

## Xem lại trận

Từ thư mục gốc AutoChessRL:

```
python -m battlesim.viewer --random 6                       # hai đội ngẫu nhiên 6 tướng
python -m battlesim.viewer --random 6 --seed 3 --stochastic # chí mạng ngẫu nhiên
python -m battlesim.viewer --a "Khiên Sắt*2@24, Cung Thủ@3" --b "Kiếm Khách@24, Pháp Sư@3"
```

Cú pháp đội: `Tên*sao@ô`, sao mặc định 1, ô nửa sân 0–27 (hàng 0 = sau, hàng 3 = đầu; ô = hàng × 7 + cột). File HTML được ghi vào `runs/replays/` rồi tự mở. Phím cách để chạy/dừng, mũi tên trái/phải để lùi/tiến một bước, bấm vào tướng để xem tầm đánh. Trong code: `from battlesim.viewer import show; show(cfg, team_a, team_b)`.

Trạng thái mỗi bước được chụp thẳng từ mảng của mô phỏng, nên trình xem lại hiện đúng những gì mô phỏng làm, kể cả lỗi. Nhãn "chặn n" dưới tướng là số bước liên tiếp nó muốn đi mà không đi được.

## Kiểm lỗi

Chạy từ thư mục gốc AutoChessRL:

```
NUMBA_DISABLE_JIT=1 python -m pytest battlesim -q     # chạy như Python thường, đặt breakpoint được
python -m pytest battlesim -q                          # bật JIT, có thêm phép so khớp JIT với Python
```

## Quy tắc giữ đối xứng

- Sát thương ghi vào mảng đệm, áp cùng lúc cuối bước.
- Hòa khoảng cách khi chọn mục tiêu: phá bằng thứ tự trong đội, không bằng chỉ số mảng.
- Tìm đường: BFS qua ô trống tới chỗ đứng đánh gần nhất theo đường đi (không theo đường chim bay); phe 1 thử hướng theo thứ tự của phe 0 xoay 180°. Mục tiêu bị vây kín thì đổi sang kẻ địch tới được bằng đường ngắn nhất.
- Chống đi qua lại: tướng cam kết với chỗ đứng đánh đã chọn tới khi nó hết dùng được; không quay về ô vừa rời trừ khi đã chờ đủ một nhịp đi (5 bước).
- Tranh ô: đường tới chỗ đánh ngắn hơn được đi; hòa thì thứ tự trong đội nhỏ hơn được đi; vẫn hòa (khác phe) thì so vị trí nhìn từ phía đội mình (phe 1 xoay 180°). Chỉ hai tướng đứng đối xứng hoàn toàn mới còn hòa; khi đó không ai đi và cả hai tránh ô đó ở bước sau.

Nhờ vậy hai đội y hệt nhau luôn ra kết quả 2, và đổi phe cho ra kết quả đối xứng chính xác (có kiểm thử).

### Các lỗi đường đi đã sửa (bắt được nhờ trình xem lại)

1. Luật "hòa thì không ai đi": hai đồng đội cùng muốn một ô chờ nhau tới hết giờ.
2. Đi theo đường chim bay một bước: tướng kẹt sau đồng đội đi qua lại giữa hai ô cùng khoảng cách thay vì lùi một ô rồi vòng ra sau (trận người dùng báo 04/11: Thích Khách 1 sao qua lại 13 lần; sau khi sửa ra đòn ở bước 31 thay vì ~130, và đổi luôn đội thắng).
3. Hai tướng địch đối diện cùng muốn ô giữa, hòa mãi; và tướng lật đường qua lại khi đồng đội xung quanh xê dịch.

Kiểm thử bất biến giờ đếm hai dấu hiệu kẹt: có đường tới chỗ đánh mà 10 bước liền không đi, và đi về ô vừa rời quá 3 lần liên tiếp mà chưa ra đòn. Trên 5.000 trận ngẫu nhiên không tướng nào quay đầu quá 1 lần liên tiếp; tỉ lệ hết giờ (8 đấu 8) khoảng 1,8%.

## Khi lên PC

`_run_all` đã khai báo `parallel=True` và dùng `prange` theo trận: chỉ cần tăng B. Trên máy thử 2 nhân, 8 đấu 8 chạy khoảng 13 nghìn trận/giây.
