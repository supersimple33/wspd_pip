import wspd
import numpy as np

# Sample point set used across all tests
_DATA = [[1.0, 0.0], [2.0, 1.0], [1.0, 10.0], [2.0, 9.0],
         [10.0, 4.0], [11.0, 5.0], [5.0, 4.0], [5.5, 3.5]]
_PTS  = [wspd.point(row) for row in _DATA]
_N, _DIM, _S = len(_PTS), 2, 2.0

# --- build_wspd (baseline) ---
_dumbbells = wspd.build_wspd(_N, _DIM, _S, _PTS)
assert isinstance(_dumbbells, list) and len(_dumbbells) > 0

# --- build_wspd_tup_np ---
a_list, b_list = wspd.build_wspd_tup_np(_N, _DIM, _S, _PTS)
assert len(a_list) == len(_dumbbells)
assert len(b_list) == len(_dumbbells)
for i, (a, b) in enumerate(zip(a_list, b_list)):
    assert isinstance(a, np.ndarray) and a.ndim == 1 and a.dtype == np.int32
    assert isinstance(b, np.ndarray) and b.ndim == 1 and b.dtype == np.int32
    assert list(a) == list(_dumbbells[i][0])
    assert list(b) == list(_dumbbells[i][1])

# --- build_wspd_flat_np ---
a_flat, b_flat, a_off, b_off = wspd.build_wspd_flat_np(_N, _DIM, _S, _PTS)
assert isinstance(a_flat,  np.ndarray) and a_flat.ndim  == 1 and a_flat.dtype  == np.int32
assert isinstance(b_flat,  np.ndarray) and b_flat.ndim  == 1 and b_flat.dtype  == np.int32
assert isinstance(a_off,   np.ndarray) and a_off.ndim   == 1 and a_off.dtype   == np.int32
assert isinstance(b_off,   np.ndarray) and b_off.ndim   == 1 and b_off.dtype   == np.int32
assert len(a_off) == len(_dumbbells)
assert len(b_off) == len(_dumbbells)
# offsets must be non-decreasing and start at 0
assert a_off[0] == 0 and b_off[0] == 0
# each pair's slice in the flat arrays matches the tup_np result
for i, (a, b) in enumerate(zip(a_list, b_list)):
    a_start = a_off[i]
    a_end   = a_off[i + 1] if i + 1 < len(a_off) else len(a_flat)
    b_start = b_off[i]
    b_end   = b_off[i + 1] if i + 1 < len(b_off) else len(b_flat)
    assert np.array_equal(a_flat[a_start:a_end], a)
    assert np.array_equal(b_flat[b_start:b_end], b)

