"""约束回归测试：参考拟合一致性、单调性、逆映射、错误分类、性能、并发。"""

import math
import threading
import time

import pytest

import aces

A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14


def _reference(x):
    """标准 Narkowicz ACES filmic 拟合。"""
    return (x * (A * x + B)) / (x * (C * x + D) + E)


def _clamped_reference(x):
    return min(_reference(x), 1.0)


# --- 关键性质 ---------------------------------------------------------------
def test_zero_maps_to_zero():
    assert aces.Aces().curve(0.0) == 0.0


def test_infinity_limit_is_one():
    a = aces.Aces()
    assert a.curve(1e6) <= 1.0
    assert a.curve(1e308) == 1.0
    # 大值区间必须严格递增并收敛到 1
    assert a.curve(1e6) < a.curve(1e9) <= 1.0


def test_midtone_matches_reference_exactly():
    a = aces.Aces()
    for i in range(1, 2000):
        x = i * a.X0 / 2000
        assert a.curve(x) == _reference(x)


def test_full_range_within_tolerance():
    a = aces.Aces()
    for i in range(0, 20001):
        x = i * 1e4 / 20000
        assert abs(a.curve(x) - _clamped_reference(x)) <= 1e-3


def test_curve_strictly_monotonic():
    a = aces.Aces()
    prev = a.curve(0.0)
    for i in range(1, 100001):
        v = a.curve(i * 10.0 / 100000)
        assert v > prev
        prev = v


def test_curve_range_and_float_type():
    a = aces.Aces()
    for i in range(0, 1001):
        v = a.curve(i * 10.0 / 1000)
        assert isinstance(v, float)
        assert 0.0 <= v < 1.0


def test_seam_continuity():
    a = aces.Aces()
    lo = math.nextafter(a.X0, -math.inf)
    hi = math.nextafter(a.X0, math.inf)
    assert abs(a.curve(lo) - a.curve(hi)) < 1e-12


# --- 逆映射 -----------------------------------------------------------------
def test_inverse_roundtrip_relative():
    a = aces.Aces()
    for i in range(1, 20001):
        x = 10 ** (-3 + 7 * i / 20000)  # 1e-3 .. 1e4
        y = a.curve(x)
        assert abs(a.inverse(y) - x) / x <= 1e-3


def test_inverse_roundtrip_absolute():
    a = aces.Aces()
    for i in range(0, 10001):
        x = i * 10.0 / 10000
        y = a.curve(x)
        assert abs(a.inverse(y) - x) <= 1e-3


def test_inverse_roundtrip_far_highlights():
    # float64 的 y 在接近 1 时分辨率有限；在 x <= 1e5 内绝对误差仍远小于 1e-3。
    a = aces.Aces()
    for k in range(0, 2001):
        x = 10 ** (k * 5.0 / 2000)
        assert abs(a.inverse(a.curve(x)) - x) <= 1e-3


def test_inverse_endpoints():
    a = aces.Aces()
    assert a.inverse(0.0) == 0.0
    assert a.inverse(1.0) == math.inf


def test_inverse_monotonic():
    a = aces.Aces()
    prev = a.inverse(0.0)
    for i in range(1, 10000):
        v = a.inverse(i / 10000)
        assert v > prev
        prev = v


def test_inverse_of_midtone_is_exact():
    a = aces.Aces()
    for i in range(1, 1000):
        x = i * a.X0 / 1000
        assert abs(a.inverse(_reference(x)) - x) <= 1e-9 * x


# --- 错误分类 ---------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,x",
    [
        (aces.NaNInputError, float("nan")),
        (aces.PositiveInfinityInputError, float("inf")),
        (aces.NegativeInfinityInputError, float("-inf")),
        (aces.NegativeInputError, -0.1),
        (aces.InputTypeError, "0.3"),
        (aces.InputTypeError, None),
        (aces.InputTypeError, [1.0]),
    ],
)
def test_curve_error_classes(exc, x):
    with pytest.raises(exc):
        aces.Aces().curve(x)
    # 全部仍然可被 ValueError 捕获（类型错误另含 TypeError）
    with pytest.raises(ValueError):
        aces.Aces().curve(x)


@pytest.mark.parametrize(
    "exc,y",
    [
        (aces.NaNInputError, float("nan")),
        (aces.PositiveInfinityInputError, float("inf")),
        (aces.NegativeInfinityInputError, float("-inf")),
        (aces.NegativeLuminanceError, -0.1),
        (aces.LuminanceOutOfRangeError, 1.5),
        (aces.InputTypeError, "0.5"),
    ],
)
def test_inverse_error_classes(exc, y):
    with pytest.raises(exc):
        aces.Aces().inverse(y)
    with pytest.raises(ValueError):
        aces.Aces().inverse(y)


def test_error_classes_are_distinguishable():
    errors = {}
    for value in (float("nan"), float("inf"), float("-inf"), -1.0):
        try:
            aces.Aces().curve(value)
        except ValueError as exc:
            errors[type(exc)] = value
    assert len(errors) == 4


# --- 性能与并发 -------------------------------------------------------------
def test_performance_100k_under_100ms():
    a = aces.Aces()
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        for i in range(100000):
            a.curve(i * 1e-5)
        best = min(best, time.perf_counter() - start)
    assert best < 0.100, "100k curve() calls took %.1f ms" % (best * 1e3)


def test_performance_inverse_100k_under_100ms():
    a = aces.Aces()
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        for i in range(100000):
            a.inverse(i * 1e-5)
        best = min(best, time.perf_counter() - start)
    assert best < 0.100, "100k inverse() calls took %.1f ms" % (best * 1e3)


def test_concurrent_calls_are_consistent():
    a = aces.Aces()
    xs = [i * 1e-3 for i in range(500)]
    expected = [(a.curve(x), a.inverse(a.curve(x))) for x in xs]
    failures = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(50):
            for x, want in zip(xs, expected):
                got = (a.curve(x), a.inverse(a.curve(x)))
                if got != want:
                    failures.append((x, got, want))
                    return

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == []


def test_shared_module_functions():
    assert isinstance(aces.curve(0.18), float)
    assert abs(aces.inverse(aces.curve(0.5)) - 0.5) <= 1e-3
