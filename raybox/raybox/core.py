"""射线与轴对齐包围盒（AABB）求交，数值稳健实现。

设计要点
--------
* **slab 法**：每个轴只做一次除法，取各轴进入/离开参数的 ``max``/``min``。
  这与逐面求交在浮点运算上等价（同样的 ``(bound - origin) / dir``），
  因此两种实现的结果偏差为 0（远小于要求的 1e-9）。
* **起点在盒内**：进入参数小于 0 时归零，返回 ``(0.0, t_exit)``。
* **平行轴**：方向分量为 0 时不做除法；起点落在该 slab 之外即判为不命中。
  起点恰好落在边界上（graze）按闭区间处理。
* **错误分类**：方向全零与方向含 NaN/Inf 抛出*不同*的异常类型，且都继承
  ``ValueError``；``ZeroDirectionError`` 同时继承 ``ZeroDivisionError``，
  以兼顾“除零”这一语义。非有限输入按 NaN / Inf 进一步细分。
* **并发安全**：``RayBox`` 不含任何实例或类级别的可变状态，求交过程只使用
  局部变量，实例可被多个线程同时复用；入参也不会被修改。
"""

import math

__all__ = [
    "RayBox",
    "intersect",
    "hit",
    "RayBoxError",
    "InvalidInputError",
    "NonFiniteError",
    "NaNInputError",
    "InfiniteInputError",
    "ZeroDirectionError",
    "DegenerateBoxError",
    "ShapeError",
]

_INF = float("inf")
_isfinite = math.isfinite


# --------------------------------------------------------------------------- #
# 异常层级：全部继承 ValueError，便于统一捕获；子类之间互不兼容，便于区分。
# --------------------------------------------------------------------------- #
class RayBoxError(ValueError):
    """raybox 所有输入错误的基类。"""


class InvalidInputError(RayBoxError):
    """输入类型或形状非法。"""


class ShapeError(InvalidInputError):
    """输入形状非法（例如不是三元组 / 盒子的 min/max 结构不对）。"""


class NonFiniteError(InvalidInputError):
    """输入含非有限浮点数（NaN 或 ±Inf）。"""


class NaNInputError(NonFiniteError):
    """输入含 NaN。"""


class InfiniteInputError(NonFiniteError):
    """输入含 ±Inf（或整型无法表示为有限 float）。"""


class ZeroDirectionError(InvalidInputError, ZeroDivisionError):
    """方向向量三个分量全为 0。

    同时是 ``ValueError`` 与 ``ZeroDivisionError``，从而既能按“非法输入”也能
    按“除零”语义捕获；与 :class:`NaNInputError` 类型不同，可明确区分。
    """


class DegenerateBoxError(InvalidInputError):
    """盒子的最小值大于最大值（体积为负，非法）。"""


# --------------------------------------------------------------------------- #
# 输入解析与校验（无状态，只读入参）
# --------------------------------------------------------------------------- #
def _raise_non_finite(name, values):
    for index, value in enumerate(values):
        if not _isfinite(value):
            if value != value:
                raise NaNInputError(f"{name}[{index}] is NaN")
            raise InfiniteInputError(f"{name}[{index}] is infinite")
    raise NonFiniteError(f"{name} contains a non-finite value: {values!r}")


def _triple(value, name):
    """把长度为 3 的序列转成 ``(float, float, float)``。"""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ShapeError(f"{name} must have exactly 3 components, got {len(value)}")
        a, b, c = value
    else:
        try:
            a, b, c = value
        except (TypeError, ValueError):
            raise ShapeError(f"{name} must be a sequence of 3 numbers") from None

    try:
        return float(a), float(b), float(c)
    except OverflowError:
        # 数值过大（如 10**400）无法表示为有限 float。
        for index, item in enumerate((a, b, c)):
            try:
                float(item)
            except OverflowError:
                raise InfiniteInputError(f"{name}[{index}] overflows float") from None
            except (TypeError, ValueError):
                raise InvalidInputError(f"{name}[{index}] is not a real number: {item!r}") from None
        raise InfiniteInputError(f"{name} overflows float") from None
    except (TypeError, ValueError):
        for index, item in enumerate((a, b, c)):
            try:
                float(item)
            except OverflowError:
                raise InfiniteInputError(f"{name}[{index}] overflows float") from None
            except (TypeError, ValueError):
                raise InvalidInputError(f"{name}[{index}] is not a real number: {item!r}") from None
        raise InvalidInputError(f"{name} must contain real numbers") from None


def _box(value):
    """解析 ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` 并校验非退化。"""
    try:
        lo_raw, hi_raw = value
    except (TypeError, ValueError):
        raise ShapeError(
            "box must be ((xmin, ymin, zmin), (xmax, ymax, zmax))"
        ) from None

    lo = _triple(lo_raw, "box.min")
    hi = _triple(hi_raw, "box.max")

    if not (
        _isfinite(lo[0]) and _isfinite(lo[1]) and _isfinite(lo[2])
    ):
        _raise_non_finite("box.min", lo)
    if not (
        _isfinite(hi[0]) and _isfinite(hi[1]) and _isfinite(hi[2])
    ):
        _raise_non_finite("box.max", hi)

    if lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
        raise DegenerateBoxError(
            f"box.min must not exceed box.max: min={lo!r}, max={hi!r}"
        )
    return lo, hi


def _validate_origin(origin):
    o = _triple(origin, "origin")
    if not (_isfinite(o[0]) and _isfinite(o[1]) and _isfinite(o[2])):
        _raise_non_finite("origin", o)
    return o


def _validate_direction(direction):
    d = _triple(direction, "direction")
    dx, dy, dz = d
    if not (_isfinite(dx) and _isfinite(dy) and _isfinite(dz)):
        _raise_non_finite("direction", d)
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        raise ZeroDirectionError("direction is the zero vector; cannot define a ray")
    return d


def _intersect(origin, direction, box):
    ox, oy, oz = _validate_origin(origin)
    dx, dy, dz = _validate_direction(direction)
    lo, hi = _box(box)
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    tmin = -_INF
    tmax = _INF

    # --- x slab ---
    if dx != 0.0:
        if dx > 0.0:
            near = (x0 - ox) / dx
            far = (x1 - ox) / dx
        else:
            near = (x1 - ox) / dx
            far = (x0 - ox) / dx
        if near > tmin:
            tmin = near
        if far < tmax:
            tmax = far
        if tmin > tmax:
            return None
    elif ox < x0 or ox > x1:
        return None

    # --- y slab ---
    if dy != 0.0:
        if dy > 0.0:
            near = (y0 - oy) / dy
            far = (y1 - oy) / dy
        else:
            near = (y1 - oy) / dy
            far = (y0 - oy) / dy
        if near > tmin:
            tmin = near
        if far < tmax:
            tmax = far
        if tmin > tmax:
            return None
    elif oy < y0 or oy > y1:
        return None

    # --- z slab ---
    if dz != 0.0:
        if dz > 0.0:
            near = (z0 - oz) / dz
            far = (z1 - oz) / dz
        else:
            near = (z1 - oz) / dz
            far = (z0 - oz) / dz
        if near > tmin:
            tmin = near
        if far < tmax:
            tmax = far
        if tmin > tmax:
            return None
    elif oz < z0 or oz > z1:
        return None

    # 射线只覆盖 t >= 0：完全在背后的交集没有意义。
    # tmax == 0 表示起点恰在盒子表面且方向朝外——只在 t = 0 处“擦到”，
    # 前向交集为空，与起点在表面朝外的语义保持一致，判为未命中。
    if tmax <= 0.0:
        return None
    # 起点在盒内（进入参数为负）时，把进入参数归零。
    if tmin <= 0.0:
        tmin = 0.0
    return (tmin, tmax)


# --------------------------------------------------------------------------- #
# 公共 API
# --------------------------------------------------------------------------- #
class RayBox:
    """射线 - AABB 求交器。

    无状态：可安全地创建一次并在多个线程间共享。
    """

    __slots__ = ()

    def intersect(self, origin, direction, box):
        """求交，返回 ``(tmin, tmax)``（均为 float，沿射线方向的距离参数）；
        未命中返回 ``None``。

        若起点位于盒内，``tmin`` 为 ``0.0``，``tmax`` 为出射距离。
        方向分量为 0 的轴按平行轴处理。
        """
        return _intersect(origin, direction, box)

    def hit(self, origin, direction, box):
        """是否命中，返回 ``True`` / ``False``。"""
        return _intersect(origin, direction, box) is not None


def intersect(origin, direction, box):
    """无状态便捷函数，等价于 ``RayBox().intersect(...)``。"""
    return _intersect(origin, direction, box)


def hit(origin, direction, box):
    """无状态便捷函数，等价于 ``RayBox().hit(...)``。"""
    return _intersect(origin, direction, box) is not None
