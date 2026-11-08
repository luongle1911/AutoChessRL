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
