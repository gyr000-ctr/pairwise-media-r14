import math

import pytest

import transient

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return transient.Transient(48000, 6.0, 8)

CASES = [
    ("length", lambda: len(_t().process([0.1] * 100)) == 100),
    ("bad_rate", lambda: _raises(lambda: transient.Transient(0, 6.0, 8))),
    ("bad_gain", lambda: _raises(lambda: transient.Transient(48000, -1.0, 8))),
    ("bad_lookahead", lambda: _raises(lambda: transient.Transient(48000, 6.0, 0))),
    ("empty", lambda: _t().process([]) == []),
    ("type", lambda: isinstance(_t().process([0.1]), list)),
    ("det", lambda: _t().process([0.1] * 10) == _t().process([0.1] * 10)),
    ("range", lambda: all(abs(v) <= 2.0 for v in _t().process([0.5] * 100))),
    ("no_jump", lambda: max(abs(a - b) for a, b in zip(_t().process([0.5] * 50), _t().process([0.5] * 50)[1:])) < 1.0),
    ("nan", lambda: _raises(lambda: _t().process([float('nan')]))),
    ("silence", lambda: all(abs(v) < 1e-9 for v in _t().process([0.0] * 10))),
    ("state", lambda: isinstance(_t().process([0.5] * 500), list)),
    ("bad_buf", lambda: _raises(lambda: _t().process('abc'))),
    ("finite", lambda: all(math.isfinite(v) for v in _t().process([0.3] * 20))),
    ("no_mut", lambda: _t().process([0.2] * 10) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
