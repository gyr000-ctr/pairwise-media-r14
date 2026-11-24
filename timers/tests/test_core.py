import math

import pytest

import timers

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return timers.TimerWheel(8)

CASES = [
    ("empty_tick", lambda: _t().tick(1.0) == []),
    ("fire", lambda: (_t().add(1.0, 'a'), _t().tick(1.0))[1] == ['a']),
    ("order", lambda: (_t().add(1.0, 'a'), _t().add(1.0, 'b'), _t().tick(1.0))[2] == ['a', 'b']),
    ("cancel", lambda: (_t().add(1.0, 'a'), _t().cancel('a'), _t().tick(1.0))[2] == []),
    ("bad_slots", lambda: _raises(lambda: timers.TimerWheel(0))),
    ("bad_delay", lambda: _raises(lambda: _t().add(-1.0, 'a'))),
    ("big_dt", lambda: (_t().add(100.0, 'a'), _t().tick(200.0))[1] == ['a']),
    ("type", lambda: isinstance(_t().tick(1.0), list)),
    ("det", lambda: (_t().add(2.0, 'x'), _t().tick(2.0))[1] == ['x']),
    ("cancel_missing", lambda: _raises(lambda: _t().cancel('zz'))),
    ("dup_name", lambda: _raises(lambda: (_t().add(1.0, 'a'), _t().add(1.0, 'a')))),
    ("no_early", lambda: (_t().add(5.0, 'a'), _t().tick(1.0))[1] == []),
    ("many", lambda: len([_t().add(1.0, str(i)) for i in range(3)]) == 3),
    ("zero_dt", lambda: _t().tick(0.0) == []),
    ("nan_dt", lambda: _raises(lambda: _t().tick(float('nan')))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
