import math

import pytest

import aces

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return aces.Aces()

CASES = [
    ("zero", lambda: _a().curve(0.0) == 0.0),
    ("bounded", lambda: _a().curve(100.0) <= 1.0),
    ("monotonic", lambda: _a().curve(0.5) <= _a().curve(1.0)),
    ("inverse_roundtrip", lambda: abs(_a().inverse(_a().curve(0.5)) - 0.5) < 1e-3),
    ("mid", lambda: 0.0 < _a().curve(0.18) < 0.5),
    ("bad_negative", lambda: _raises(lambda: _a().curve(-0.1))),
    ("bad_inverse", lambda: _raises(lambda: _a().inverse(1.5))),
    ("type", lambda: isinstance(_a().curve(0.3), float)),
    ("det", lambda: _a().curve(0.3) == _a().curve(0.3)),
    ("nan", lambda: _raises(lambda: _a().curve(float('nan')))),
    ("inverse_zero", lambda: abs(_a().inverse(0.0)) < 1e-6),
    ("curve_range", lambda: all(0.0 <= _a().curve(x / 10.0) <= 1.0 for x in range(11))),
    ("inverse_monotonic", lambda: _a().inverse(0.2) <= _a().inverse(0.8)),
    ("bad_inverse_neg", lambda: _raises(lambda: _a().inverse(-0.1))),
    ("no_mut", lambda: _a().curve(0.4) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
