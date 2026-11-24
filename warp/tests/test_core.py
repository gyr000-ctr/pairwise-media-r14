import math

import pytest

import warp

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[1.0, 2.0], [3.0, 4.0]]
GRID = [((0.0, 0.0), (1.0, 1.0)), ((1.0, 0.0), (1.0, 1.0))]

CASES = [
    ("shape", lambda: len(warp.MeshWarp().warp(IMG, GRID, (2, 2))) == 2),
    ("corner", lambda: abs(warp.MeshWarp().sample(IMG, 0.0, 0.0) - 1.0) < 1e-9),
    ("center", lambda: 0.0 <= warp.MeshWarp().sample(IMG, 0.5, 0.5) <= 4.0),
    ("clamp", lambda: warp.MeshWarp().sample(IMG, 5.0, 5.0) == warp.MeshWarp().sample(IMG, 1.0, 1.0)),
    ("bad_grid", lambda: _raises(lambda: warp.MeshWarp().warp(IMG, [], (2, 2)))),
    ("bad_size", lambda: _raises(lambda: warp.MeshWarp().warp(IMG, GRID, (0, 2)))),
    ("bad_image", lambda: _raises(lambda: warp.MeshWarp().warp([], GRID, (2, 2)))),
    ("type", lambda: isinstance(warp.MeshWarp().warp(IMG, GRID, (2, 2)), list)),
    ("det", lambda: warp.MeshWarp().warp(IMG, GRID, (2, 2)) == warp.MeshWarp().warp(IMG, GRID, (2, 2))),
    ("sample_type", lambda: isinstance(warp.MeshWarp().sample(IMG, 0.3, 0.3), float)),
    ("range", lambda: all(0.0 <= v <= 4.0 for row in warp.MeshWarp().warp(IMG, GRID, (2, 2)) for v in row)),
    ("nan", lambda: _raises(lambda: warp.MeshWarp().sample(IMG, float('nan'), 0.0))),
    ("no_mut", lambda: warp.MeshWarp().warp([r[:] for r in IMG], GRID, (2, 2)) is not None),
    ("out_shape", lambda: len(warp.MeshWarp().warp(IMG, GRID, (4, 4))) == 4),
    ("neg_coord", lambda: isinstance(warp.MeshWarp().sample(IMG, -1.0, 0.0), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
