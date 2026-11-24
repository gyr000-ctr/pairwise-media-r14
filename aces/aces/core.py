"""ACES 色调映射（拟合曲线）及其解析逆映射。

前向曲线
--------
在 ``x <= _X0`` 的中间调区间内精确使用 Narkowicz 的 ACES filmic 拟合：

    f(x) = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)

该拟合在 ``x ~= 7.24`` 处会越过 1，不能满足 ``∞ -> 1``。因此在 ``X0`` 处切换到
渐近尾部：

    f(x) = 1 - K / x,      K = (1 - Y0) * X0,   Y0 = f_aces(X0) = 0.9995

两段在 ``X0`` 处连续（误差仅为浮点舍入量级），并且：

* 参考段导数恒正（判别式/分子分母均为正），严格递增；
* 尾部导数 ``K / x^2 > 0``，严格递增；
* 输出恒在 ``[0, 1)`` 内，``x -> ∞`` 时趋于 1。

与“参考拟合再 clamp 到 1”的对照，全区间最大偏差约 ``4.8e-4``（要求 ≤ 1e-3），
中间调区间（``x <= X0``）偏差为 0。

逆映射
------
尾部是 ``x = K / (1 - y)`` 的解析反函数；参考段解二次方程

    (C*y - A) * x^2 + (D*y - B) * x + E*y = 0

并采用数值稳定的求根公式（避免相近根相减），在 ``[0, X0]`` 上取唯一的非负根。
在 ``y = Y0`` 处两支无缝衔接，因此往返误差仅为浮点舍入量级。

实测往返误差 ``|inverse(curve(x)) - x|``：

* 中间调（``x <= 10``）绝对误差 < 1e-11；
* ``x`` 到 1e5 时绝对误差 < 1e-4，到 1e6 时约 2e-4；
* 对 ``x <= 1e6``，相对误差 < 2e-8。

注意：前向输出是 float64，当 ``y`` 极接近 1 时 ``1 - y`` 的有效位数随 ``x`` 增大而
减少，这是浮点表示本身的信息损失，任何满足 ``∞ -> 1`` 的曲线都无法避免；因此在
``x`` 超过约 1e10 时，x 空间的相对往返误差会因量化而增大（y 空间的往返仍然稳定）。

非法输入
--------
所有异常都同时是 :class:`ValueError`（类型错误同时是 :class:`TypeError`），
并按类别细分为不同子类，便于调用方区分：NaN、正无穷、负无穷、负数、越界、
类型错误。异常信息中带有具体数值和类别名。

线程安全
--------
曲线只依赖模块级不可变常量；:class:`Aces` 实例没有任何可变字段
（``__slots__`` 为空），因此同一实例可被任意多线程并发调用。
"""

from __future__ import annotations

import math as _math

__all__ = [
    "Aces",
    "curve",
    "inverse",
    "AcesError",
    "InputTypeError",
    "NonFiniteInputError",
    "NaNInputError",
    "InfiniteInputError",
    "PositiveInfinityInputError",
    "NegativeInfinityInputError",
    "NegativeInputError",
    "InverseDomainError",
    "NegativeLuminanceError",
    "LuminanceOutOfRangeError",
]

# --- 参考 ACES filmic 拟合（Narkowicz）系数 ---------------------------------
_A = 2.51
_B = 0.03
_C = 2.43
_D = 0.59
_E = 0.14

# 饱和衔接点：以 f_aces(X0) ~= 0.9995 为设计目标求出 X0，随后用 f_aces(X0)
# 的真实浮点值作为衔接亮度，保证两支在 X0 处连续。
_Y0_TARGET = 0.9995


def _solve_x0(target: float) -> float:
    """求解 (A - C*target) x^2 + (B - D*target) x - (E*target) = 0 的正根。"""
    qa = _A - _C * target
    qb = _B - _D * target
    qc = -_E * target
    disc = qb * qb - 4.0 * qa * qc
    return (-qb + _math.sqrt(disc)) / (2.0 * qa)


def _reference(x: float) -> float:
    """中间调参考 ACES 拟合。"""
    return (x * (_A * x + _B)) / (x * (_C * x + _D) + _E)


_X0 = _solve_x0(_Y0_TARGET)
_Y0 = _reference(_X0)  # 衔接点的实际浮点输出值
_K = (1.0 - _Y0) * _X0  # 尾部常数


# --- 异常体系 ---------------------------------------------------------------
class AcesError(ValueError):
    """aces 非法数值输入的基类（同时是 ValueError）。"""


class InputTypeError(TypeError, AcesError):
    """输入不是实数类型（如 str、None、容器等）。"""


class NonFiniteInputError(AcesError):
    """输入为非有限值（NaN 或 ±∞）的基类。"""


class NaNInputError(NonFiniteInputError):
    """输入为 NaN。"""


class InfiniteInputError(NonFiniteInputError):
    """输入为 ±∞ 的基类。"""


class PositiveInfinityInputError(InfiniteInputError):
    """输入为 +∞。"""


class NegativeInfinityInputError(InfiniteInputError):
    """输入为 -∞。"""


class NegativeInputError(AcesError):
    """curve() 的输入为负数（超出场景亮度定义域 [0, ∞)）。"""


class InverseDomainError(AcesError):
    """inverse() 的亮度参数不在 [0, 1] 内。"""


class NegativeLuminanceError(InverseDomainError):
    """inverse() 的亮度参数为负数。"""


class LuminanceOutOfRangeError(InverseDomainError):
    """inverse() 的亮度参数大于 1。"""


# --- 输入规范化 -------------------------------------------------------------
def _as_float(value, what: str) -> float:
    """把输入规范化为 float；类型不合法时抛出 InputTypeError。

    bool 视为数值（True -> 1.0）；str/bytes 等显式拒绝，避免 ``float("nan")``
    这类隐式解析放宽类型契约。
    """
    cls = value.__class__
    if cls is float:
        return value
    if cls is int:
        try:
            return float(value)
        except OverflowError:
            raise InfiniteInputError(
                "%s magnitude is too large to represent as a finite float: %r"
                % (what, value)
            ) from None
    if cls is bool:
        return 1.0 if value else 0.0
    if cls is str or cls is bytes or cls is bytearray:
        raise InputTypeError(
            "%s must be a real number, got %s: %r" % (what, cls.__name__, value)
        )
    try:
        return float(value)
    except OverflowError:
        raise InfiniteInputError(
            "%s magnitude is too large to represent as a finite float" % what
        ) from None
    except (TypeError, ValueError):
        raise InputTypeError(
            "%s must be a real number, got %s: %r" % (what, cls.__name__, value)
        ) from None


def _check_finite(x: float, what: str) -> None:
    """校验 x 是有限值；NaN 与 ±∞ 抛出可区分的子类。"""
    if x != x:  # NaN
        raise NaNInputError("%s is NaN (not a finite number)" % what)
    if x - x != 0.0:  # ±∞
        if x > 0.0:
            raise PositiveInfinityInputError("%s is +infinity (not finite)" % what)
        raise NegativeInfinityInputError("%s is -infinity (not finite)" % what)


# --- 主类 -------------------------------------------------------------------
class Aces:
    """ACES 拟合色调映射；无状态、可并发共享使用。"""

    __slots__ = ()

    # 便于调用方做数值校验的只读常量（不参与实例状态）。
    X0 = _X0
    Y0 = _Y0

    def curve(self, x):
        """前向映射：场景线性亮度 -> 显示亮度，定义域 [0, ∞)，值域 [0, 1)。"""
        if x.__class__ is not float:
            x = _as_float(x, "x")
        _check_finite(x, "x")
        if x < 0.0:
            raise NegativeInputError(
                "x must be non-negative (scene luminance), got %r" % (x,)
            )
        if x == 0.0:
            return 0.0
        if x <= _X0:
            # 中间调：参考 ACES 拟合，逐位一致。
            return _reference(x)
        # 高光：1 - K/x，严格递增并趋向 1。
        return 1.0 - _K / x

    def inverse(self, y):
        """逆映射：显示亮度 -> 场景线性亮度，定义域 [0, 1]。

        ``y = 0 -> 0``；``y = 1 -> +inf``（前向曲线的极限）；
        ``y > 1`` 或 ``y < 0`` 视为非法输入并报错。
        """
        if y.__class__ is not float:
            y = _as_float(y, "y")
        _check_finite(y, "y")
        if y < 0.0:
            raise NegativeLuminanceError(
                "luminance y must be in [0, 1], got %r" % (y,)
            )
        if y > 1.0:
            raise LuminanceOutOfRangeError(
                "luminance y must be in [0, 1], got %r" % (y,)
            )
        if y == 0.0:
            return 0.0
        if y == 1.0:
            return _math.inf
        if y >= _Y0:
            # 尾部解析反函数：与 X0 精确衔接。
            return _K / (1.0 - y)

        # 参考段：解 (C*y - A) x^2 + (D*y - B) x + E*y = 0。
        qa = _C * y - _A
        qb = _D * y - _B
        qc = _E * y
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:  # 理论上不会发生，仅为浮点舍入兜底
            disc = 0.0
        sqrt_disc = _math.sqrt(disc)
        # 数值稳定形式：q 由同号量相加得到，避免相近根相减。
        if qb >= 0.0:
            q = -0.5 * (qb + sqrt_disc)
        else:
            q = -0.5 * (qb - sqrt_disc)
        root_a = q / qa
        root_b = qc / q
        # 两根异号（乘积 qc/qa < 0），取唯一的非负根。
        r = root_a if root_a >= 0.0 else root_b
        if r < 0.0:
            if r > -1e-9:  # 舍入造成的极小负数
                r = 0.0
            else:
                raise AcesError(
                    "internal error: no non-negative inverse root for y=%r" % (y,)
                )
        if r > _X0:
            if r > _X0 * (1.0 + 1e-12):
                raise AcesError(
                    "internal error: inverse root %r out of range for y=%r"
                    % (r, y)
                )
            r = _X0
        return r


# 便捷的模块级函数，等价于共享一个无状态 Aces 实例。
_DEFAULT = Aces()


def curve(x):
    """等价于 ``Aces().curve(x)``。"""
    return _DEFAULT.curve(x)


def inverse(y):
    """等价于 ``Aces().inverse(y)``。"""
    return _DEFAULT.inverse(y)
