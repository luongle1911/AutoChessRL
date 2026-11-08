"""Lớp bọc Numba.

- Có Numba: dùng njit/prange thật. Đặt biến môi trường NUMBA_DISABLE_JIT=1 để
  chạy như Python thường (đặt breakpoint, in giá trị được).
- Không có Numba: njit thành hàm rỗng, prange thành range, code vẫn chạy.
"""
try:
    from numba import njit, prange  # noqa: F401
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    prange = range


# Trình mô phỏng nấc 2 (sim2.py) mới viết, chưa kiểm biên dịch Numba: mặc định chạy Python thường.
# Đặt BATTLESIM_JIT2=1 để thử biên dịch (khi đã kiểm trên máy có Numba sẽ bật mặc định).
import os as _os
if HAS_NUMBA and _os.environ.get("BATTLESIM_JIT2") == "1":
    njit2, prange2 = njit, prange
else:
    def njit2(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
    prange2 = range
