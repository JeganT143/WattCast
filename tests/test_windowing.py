import numpy as np
import pytest

from src.models.windowing import make_windows


def _x(n=10, f=3, dtype="float64"):
    return np.arange(n * f, dtype=dtype).reshape(n, f)


def test_output_shape_is_n_minus_L_plus_1_by_L_by_features():
    assert make_windows(_x(10, 3), 4).shape == (7, 4, 3)


def test_every_window_is_the_matching_consecutive_slice():
    X = _x(10, 3)
    w = make_windows(X, 4)
    for i in range(len(w)):
        assert np.array_equal(w[i], X[i : i + 4]), i


def test_first_and_last_windows_are_pinned():
    X = _x(10, 3)
    w = make_windows(X, 4)
    assert np.array_equal(w[0], X[0:4])
    assert np.array_equal(w[-1], X[6:10])
    assert np.array_equal(w[-1][-1], X[-1])


def test_L_equal_to_one_gives_one_window_per_row():
    X = _x(10, 3)
    w = make_windows(X, 1)
    assert w.shape == (10, 1, 3)
    assert np.array_equal(w[:, 0, :], X)


def test_L_equal_to_n_gives_a_single_window_equal_to_the_input():
    X = _x(6, 3)
    w = make_windows(X, 6)
    assert w.shape == (1, 6, 3)
    assert np.array_equal(w[0], X)


def test_L_greater_than_n_gives_an_empty_result_with_the_right_trailing_shape():
    X = _x(3, 3)
    w = make_windows(X, 5)
    assert w.shape == (0, 5, 3)
    assert w.dtype == X.dtype


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_is_preserved(dtype):
    assert make_windows(_x(10, 3, dtype), 4).dtype == np.dtype(dtype)


def test_result_is_an_independent_writeable_copy():
    X = _x(10, 3)
    before = X.copy()
    w = make_windows(X, 4)
    assert w.flags.writeable
    w[...] = -1.0
    assert np.array_equal(X, before)


def test_result_is_c_contiguous_and_independent_of_input_layout():
    X = _x(10, 3)
    w_c = make_windows(np.ascontiguousarray(X), 4)
    w_f = make_windows(np.asfortranarray(X), 4)
    assert w_c.flags.c_contiguous and w_f.flags.c_contiguous
    assert np.array_equal(w_c, w_f)


@pytest.mark.parametrize("shape", [(10,), (2, 5, 3)])
def test_non_2d_input_raises(shape):
    with pytest.raises(ValueError, match="2-D"):
        make_windows(np.zeros(shape), 4)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_L_raises(bad):
    with pytest.raises(ValueError, match="positive"):
        make_windows(_x(), bad)


def test_a_window_depends_only_on_rows_up_to_its_last_row():
    X = _x(10, 3)
    X2 = X.copy()
    X2[7] += 1000.0
    w1, w2 = make_windows(X, 4), make_windows(X2, 4)
    # windows i cover rows i..i+3: those ending before row 7 (i <= 3) must be unchanged,
    # those containing row 7 (i = 4, 5, 6) must change
    for i in range(0, 4):
        assert np.array_equal(w1[i], w2[i]), i
    for i in range(4, 7):
        assert not np.array_equal(w1[i], w2[i]), i


@pytest.mark.parametrize("L", [1, 6])
def test_result_never_aliases_the_input_even_when_the_window_view_is_already_contiguous(L):
    X = _x(6, 3)
    before = X.copy()
    w = make_windows(X, L)
    assert w.flags.writeable
    assert not np.shares_memory(w, X)
    w[...] = -1.0
    assert np.array_equal(X, before)
