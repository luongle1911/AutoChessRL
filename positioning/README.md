# Agent xếp quân

Nhận đội mình (trên sân và hàng chờ) cùng danh sách đối thủ, chọn tướng lên sân và ô đứng sao cho xác suất thắng cao nhất, rồi trả xác suất thắng theo từng đối thủ cho agent vĩ mô.

Mô phỏng giao tranh nằm riêng ở `battlesim/`. Agent này chỉ gửi hai đội sang và nhận lại kết quả cùng thống kê theo tướng.

## Trạng thái

Mới có giao diện (`interface.py`), chưa có phần ruột. Theo kế hoạch, khi PC sửa xong sẽ sinh dữ liệu giao tranh bằng `battlesim` rồi huấn luyện mạng đánh giá đội hình. Trước đó, agent vĩ mô vẫn dùng công thức sức mạnh sẵn có trong `autochess/core/combat.py`.

## Hợp đồng với agent vĩ mô

| Hướng | Trường | Dạng | Ghi chú |
| --- | --- | --- | --- |
| Vào | Đội mình: sân + hàng chờ | (B, U) mỗi trường | mã tướng, số sao; −1 = trống |
| Vào | Đối thủ đã xếp | (B, K, U) mỗi trường | mã tướng, số sao, ô nửa sân |
| Ra | Xác suất thắng theo đối thủ | (B, K) | đánh giá lại bằng lượt mô phỏng mới, không lấy từ lượt dùng để chọn |
| Ra | Cách xếp đã chọn | (B, U) | ô nửa sân của từng tướng, −1 = để hàng chờ |

Giữ nguyên hợp đồng này thì thay ruột (công thức → mạng học được) mà agent vĩ mô không phải sửa.
