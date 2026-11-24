import math

import pytest

import raybox

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

BOX = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

CASES = [
    ("hit", lambda: raybox.RayBox().hit((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX) is True),
    ("miss", lambda: raybox.RayBox().hit((-1.0, 5.0, 0.5), (1.0, 0.0, 0.0), BOX) is False),
    ("inside", lambda: raybox.RayBox().intersect((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), BOX)[0] == 0.0),
    ("tval", lambda: abs(raybox.RayBox().intersect((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX)[0] - 1.0) < 1e-9),
    ("parallel", lambda: raybox.RayBox().hit((0.0, 0.0, 0.5), (0.0, 0.0, 1.0), BOX) is True),
    ("bad_dir", lambda: _raises(lambda: raybox.RayBox().hit((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), BOX))),
    ("bad_box", lambda: _raises(lambda: raybox.RayBox().hit((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ((1, 1, 1), (0, 0, 0))))),
    ("type", lambda: isinstance(raybox.RayBox().intersect((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX), tuple)),
    ("det", lambda: raybox.RayBox().intersect((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX) == raybox.RayBox().intersect((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX)),
    ("behind", lambda: raybox.RayBox().hit((0.0, 0.5, 0.5), (-1.0, 0.0, 0.0), BOX) is False),
    ("tmax", lambda: raybox.RayBox().intersect((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX)[1] > 1.0),
    ("nan", lambda: _raises(lambda: raybox.RayBox().hit((float('nan'), 0.0, 0.0), (1.0, 0.0, 0.0), BOX))),
    ("grazing", lambda: raybox.RayBox().hit((-1.0, 1.0, 0.5), (1.0, 0.0, 0.0), BOX) is True),
    ("neg_dir", lambda: raybox.RayBox().hit((2.0, 0.5, 0.5), (-1.0, 0.0, 0.0), BOX) is True),
    ("no_mut", lambda: raybox.RayBox().hit((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), BOX) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
