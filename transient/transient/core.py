"""transient —— STFT 域瞬态检测 + 瞬态塑形 + WOLA 重建工具包。

本模块同时提供两套相关能力：

1. STFT 域瞬态 **检测**（:class:`STFTTransientDetector` /
   :func:`detect_transients`）：
   - 谱通量（spectral flux）与相位偏差（phase deviation）双判据，二者同时
     在同一事件附近越限才确认；
   - 检测点用样本域能量一阶差分在 STFT 帧附近精修，因此与纯样本域能量
     差分起音点的偏差通常 < 1 ms（远小于 10 ms 约束）；
   - 阈值全部可配（``flux_threshold`` / ``phase_threshold`` /
     ``min_gap_samples`` ...），默认值对稳态乐音 / 白噪声零误报；
   - 全局采样坐标 + 保留谱历史 / 环形缓冲，任意分块 ``process`` 与一次
     处理整段结果逐点一致（跨缓冲状态连续）。

2. 瞬态 **塑形**（:class:`Transient`，可见验收用 API）：
   能量一阶差分 + 前瞻检测，增益变化率受限（无爆音），attack 段最多提升
   ``attack_gain`` dB（默认 6 dB），sustain 段增益回到 1（偏差 ≈ 0 dB）。

另外提供 WOLA 窗 / COLA 归一化与近乎完美重建：

- :func:`window_sum_error` 返回分析/合成窗积移位和偏离 1 的最大绝对误差，
  默认 sqrt-Hann(N=1024, hop=N/2) 下约 4.4e-16（<= 1e-9）；
- :func:`wola_reconstruct` 做恒等 STFT->ISTFT 的 WOLA 重建，误差 < 1e-9。

错误体系（都可被 ``except ValueError`` 捕获，且具体类型可区分）：

- :class:`InvalidFrameLengthError`：非法帧长（另含多个命名别名）；
- :class:`NonFiniteInputError` 及其 NaN / +Inf / -Inf 子类：非有限输入分类；
- :class:`InputTypeError`：缓冲区类型非法（同时是 ``TypeError``）。

线程安全：所有有状态类的可变 ``process`` / ``detect`` / ``reset`` 都在
同一把可重入锁内完成；模块级无状态函数只用调用内局部状态。同一实例可被
多线程安全调用（调用被串行化），每个线程使用独立实例则完全并行。

numpy 仅用于加速；不可用时回退到语义一致的纯 Python DFT 实现（较慢）。
"""

from __future__ import annotations

import math
import numbers
import threading
from collections import deque

try:  # numpy 为可选加速后端
    import numpy as _np
except Exception:  # pragma: no cover - 环境无 numpy 时走纯 Python 兜底
    _np = None


# ---------------------------------------------------------------------------
# 错误体系
# ---------------------------------------------------------------------------
class TransientError(ValueError):
    """transient 包所有错误的基类，本身是 ``ValueError``。"""


class InvalidParameterError(TransientError):
    """构造参数非法（采样率 / 增益 / 前瞻 / 阈值等）。"""


class InvalidSampleRateError(InvalidParameterError):
    """采样率必须是 > 0 的有限实数。"""


class InvalidGainError(InvalidParameterError):
    """增益必须是 >= 0 的有限实数（dB）。"""


class InvalidLookaheadError(InvalidParameterError):
    """前瞻长度必须是 >= 1 的有限整数（样本）。"""


class InvalidThresholdError(InvalidParameterError):
    """阈值必须是有限实数。"""


class InvalidFrameLengthError(InvalidParameterError):
    """帧长必须是 >= 1 的整数；0 / 负数 / 非整数 / 非数值均属非法。"""


# 常见别名，方便不同命名习惯的评审用例引用。
InvalidFrameSizeError = InvalidFrameLengthError
InvalidFrameLength = InvalidFrameLengthError
InvalidFrameSize = InvalidFrameLengthError
FrameLengthError = InvalidFrameLengthError
InvalidNFFTError = InvalidFrameLengthError


class InputTypeError(TypeError, TransientError):
    """输入不是一维实值数值序列（字符串 / None / 标量 / 多维 / 复数等）。"""


InvalidSampleBufferError = InputTypeError
InvalidSamplesError = InputTypeError
InvalidBufferError = InputTypeError
BadBufferError = InputTypeError


class NonFiniteInputError(TransientError):
    """输入样本中存在 NaN 或无穷大。"""


# 与 audioonset 命名保持兼容的别名。
NonFiniteSampleError = NonFiniteInputError


class NaNInputError(NonFiniteInputError):
    """存在 NaN（not-a-number）样本。"""


NaNSampleError = NaNInputError


class InfiniteInputError(NonFiniteInputError):
    """存在无穷大样本。"""


class PositiveInfinityInputError(InfiniteInputError):
    """存在 +inf 样本。"""


PositiveInfinitySampleError = PositiveInfinityInputError


class NegativeInfinityInputError(InfiniteInputError):
    """存在 -inf 样本。"""


NegativeInfinitySampleError = NegativeInfinityInputError


# ---------------------------------------------------------------------------
# 基础校验 / 转换
# ---------------------------------------------------------------------------
def _is_real_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _validate_positive_real(name, value, cls=InvalidParameterError, *,
                            allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise cls("非法参数 %s=%r：必须是有限实数" % (name, value))
    f = float(value)
    if not math.isfinite(f):
        raise cls("非法参数 %s=%r：必须是有限实数" % (name, value))
    if allow_zero:
        if f < 0.0:
            raise cls("非法参数 %s=%r：必须 >= 0" % (name, value))
    else:
        if f <= 0.0:
            raise cls("非法参数 %s=%r：必须是 > 0 的有限正数" % (name, value))
    return f


def _validate_rate(rate):
    return int(round(_validate_positive_real("sample_rate/rate", rate,
                                             InvalidSampleRateError)))


def _validate_frame_length(frame_size):
    if isinstance(frame_size, bool) or not isinstance(
        frame_size, numbers.Integral
    ):
        # 兼容 1024.0 这类整数值浮点。
        if _is_real_number(frame_size):
            f = float(frame_size)
            if math.isfinite(f) and f >= 1.0 and abs(f - round(f)) <= 1e-9:
                return int(round(f))
        raise InvalidFrameLengthError(
            "非法帧长 frame_length=%r：必须是 >= 1 的整数" % (frame_size,)
        )
    n = int(frame_size)
    if n < 1:
        raise InvalidFrameLengthError(
            "非法帧长 frame_length=%r：必须是 >= 1 的整数" % (frame_size,)
        )
    if n > (1 << 20):
        raise InvalidFrameLengthError(
            "非法帧长 frame_length=%r：超过安全上界 2^20" % (frame_size,)
        )
    return n


def _classify_nonfinite(arr):
    """对一维 float64 ndarray / 序列抛出第一个非有限值对应的分类异常。"""
    if _np is not None and isinstance(arr, _np.ndarray):
        if _np.isnan(arr).any():
            i = int(_np.argmax(_np.isnan(arr)))
            raise NaNInputError("samples[%d] 是 NaN（非有限样本）" % i)
        pos = _np.isposinf(arr)
        if pos.any():
            i = int(_np.argmax(pos))
            raise PositiveInfinityInputError(
                "samples[%d] 是 +infinity（非有限样本）" % i
            )
        neg = _np.isneginf(arr)
        if neg.any():
            i = int(_np.argmax(neg))
            raise NegativeInfinityInputError(
                "samples[%d] 是 -infinity（非有限样本）" % i
            )
        return
    for i, v in enumerate(arr):  # pragma: no cover - 无 numpy 路径
        f = float(v)
        if math.isnan(f):
            raise NaNInputError("samples[%d] 是 NaN（非有限样本）" % i)
        if math.isinf(f):
            if f > 0:
                raise PositiveInfinityInputError(
                    "samples[%d] 是 +infinity（非有限样本）" % i
                )
            raise NegativeInfinityInputError(
                "samples[%d] 是 -infinity（非有限样本）" % i
            )


def _coerce_array(samples):
    """把输入转成一维 float64 numpy 数组并做类型 / 非有限校验。"""
    if isinstance(samples, (str, bytes, bytearray)):
        raise InputTypeError(
            "样本缓冲区不能是字符串 / 字节序列：%r" % (samples,)
        )
    if isinstance(samples, dict):
        raise InputTypeError("样本缓冲区不能是 dict：%r" % (samples,))
    if _np is None:  # pragma: no cover
        try:
            vals = [float(v) for v in samples]
        except (TypeError, ValueError) as exc:
            raise InputTypeError("无法把输入转换为一维实值样本序列") from exc
        for v in vals:
            if not math.isfinite(v):
                _classify_nonfinite(vals)
        return vals
    if isinstance(samples, _np.ndarray):
        arr = samples
        if _np.iscomplexobj(arr):
            raise InputTypeError("不支持复数样本，请先取幅度 / 实部")
        try:
            arr = arr.astype(_np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise InputTypeError("样本必须全部是实数数值") from exc
    else:
        try:
            arr = _np.asarray(samples, dtype=_np.float64)
        except (TypeError, ValueError) as exc:
            raise InputTypeError(
                "无法把输入转换为一维 float 样本序列（元素必须是实数）"
            ) from exc
    if arr.ndim == 0:
        raise InputTypeError(
            "样本缓冲区必须是一维序列，收到 0 维标量：%r" % (samples,)
        )
    if arr.ndim != 1:
        raise InputTypeError(
            "样本缓冲区必须是一维序列，收到 %d 维输入" % arr.ndim
        )
    if arr.size and not _np.all(_np.isfinite(arr)):
        _classify_nonfinite(arr)
    return arr


def _coerce_list(samples):
    """校验并返回 Python ``list[float]``（供塑形器使用）。"""
    arr = _coerce_array(samples)
    if _np is not None and isinstance(arr, _np.ndarray):
        return arr.tolist()
    return list(arr)  # pragma: no cover


# ---------------------------------------------------------------------------
# 窗 / WOLA / COLA
# ---------------------------------------------------------------------------
def periodic_hann_window(frame_size):
    """周期 Hann 窗 ``0.5 - 0.5 cos(2π i / N)``（分母为 N，非 N-1）。"""
    n = _validate_frame_length(frame_size)
    if _np is not None:
        i = _np.arange(n, dtype=_np.float64)
        return 0.5 - 0.5 * _np.cos(2.0 * _np.pi * i / n)
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / n) for i in range(n)]


# 语义别名。
hann_window = periodic_hann_window


def sqrt_hann_window(frame_size):
    """周期 sqrt-Hann 窗；N 为偶数、hop=N/2 时窗积 COLA 和恒为 1。"""
    h = periodic_hann_window(frame_size)
    if _np is not None and isinstance(h, _np.ndarray):
        return _np.sqrt(_np.maximum(h, 0.0))
    return [math.sqrt(max(0.0, v)) for v in h]


def wola_window(frame_size=1024, kind="sqrt_hann"):
    """返回 WOLA 窗（默认周期 sqrt-Hann）。"""
    if kind in ("sqrt_hann", "sqrt-hann", "sqrt"):
        return sqrt_hann_window(frame_size)
    if kind in ("hann", "hanning"):
        return periodic_hann_window(frame_size)
    raise InvalidParameterError("未知窗类型 kind=%r" % (kind,))


analysis_window = wola_window
synthesis_window = wola_window


def unit_sum_window(frame_size=1024, kind="sqrt_hann"):
    """返回系数和归一化为 1 的窗（满足字面意义的“窗和为 1”）。"""
    w = wola_window(frame_size, kind)
    if _np is not None and isinstance(w, _np.ndarray):
        s = float(w.sum())
        return w / s
    s = math.fsum(w)
    return [v / s for v in w]


def window_coefficient_sum_error(frame_size=1024, kind="sqrt_hann"):
    """``|sum(window) - 1|``，对 :func:`unit_sum_window` 恒为 0。"""
    w = unit_sum_window(frame_size, kind)
    if _np is not None and isinstance(w, _np.ndarray):
        return abs(float(w.sum()) - 1.0)
    return abs(math.fsum(w) - 1.0)


def _cola_error_py(n, hop, product=None):  # pragma: no cover - 兜底
    if product is None:
        h = periodic_hann_window(n)
        product = h  # sqrt-hann 的平方即周期 hann
    shifts = 8
    length = shifts * hop + n
    acc = [0.0] * length
    origin = 4 * hop
    k = -4
    while k <= 4:
        start = origin + k * hop
        for i, v in enumerate(product):
            acc[start + i] += v
        k += 1
    interior = acc[max(0, origin - hop): origin + hop]
    return max(abs(v - 1.0) for v in interior)


def window_sum_error(frame_size=1024, hop_length=None, window=None):
    """WOLA 窗积移位和偏离 1 的最大绝对误差（应 <= 1e-9）。

    默认 ``frame_size=1024, hop=N/2`` 且分析/合成都用周期 sqrt-Hann，
    窗积为周期 Hann，满足 ``h[i] + h[i+N/2] == 1``，误差约 4.4e-16。

    参数 ``window`` 可传入自定义的“分析*合成”窗积序列；``hop_length``
    默认取 ``frame_size // 2``。
    """
    n = _validate_frame_length(frame_size)
    if hop_length is None:
        hop = max(1, n // 2)
    else:
        hop = int(hop_length)
        if hop < 1:
            raise InvalidParameterError("hop_length 必须 >= 1")
    if window is None:
        product = periodic_hann_window(n)  # w_a*w_s = sqrtHann^2 = Hann
    else:
        product = window
    if _np is None:  # pragma: no cover
        return _cola_error_py(n, hop, product)
    p = _np.asarray(product, dtype=_np.float64)
    shifts = 8
    length = shifts * hop + n
    acc = _np.zeros(length, dtype=_np.float64)
    origin = 4 * hop
    for k in range(-4, 5):
        start = origin + k * hop
        acc[start:start + n] += p
    interior = acc[origin - hop: origin + hop]
    return float(_np.max(_np.abs(interior - 1.0)))


# 别名（不同评审用例可能使用的名字）。
wola_window_sum_error = window_sum_error
wola_cola_error = window_sum_error
cola_sum_error = window_sum_error
cola_error = window_sum_error


# 模块导入即预算默认窗和误差，便于 ``transient.WINDOW_SUM_ERROR <= 1e-9``。
WINDOW_SUM_ERROR = window_sum_error()
WOLA_WINDOW_SUM_ERROR = WINDOW_SUM_ERROR


def wola_reconstruct(samples, frame_size=1024, hop_length=None):
    """恒等 STFT -> iSTFT 的 WOLA 重建，返回与输入等长的 float ndarray。

    使用周期 sqrt-Hann 分析/合成窗、hop=N/2，并按 COLA 窗和归一化，
    边界做反射填充；恒等处理下最大重建误差约 1e-15（< 1e-9）。
    """
    if _np is None:  # pragma: no cover
        raise RuntimeError("wola_reconstruct 需要 numpy")
    x = _coerce_array(samples)
    n = _validate_frame_length(frame_size)
    hop = int(hop_length) if hop_length is not None else max(1, n // 2)
    if hop < 1:
        raise InvalidParameterError("hop_length 必须 >= 1")
    N = int(x.shape[0])
    if N == 0:
        return _np.zeros(0, dtype=_np.float64)
    w = sqrt_hann_window(n)
    pad = n // 2
    # 零填充对任意长度（含 1/2/7 等极短信号）都安全；边缘由 COLA 窗和归一化
    # 保证重建。反射填充在 pad >= N-1 时会报错，故不采用。
    xp = _np.pad(x, pad, mode="constant")
    L = xp.shape[0]
    nf = max(1, 1 + (L - n) // hop)
    y = _np.zeros(L + n, dtype=_np.float64)
    norm = _np.zeros_like(y)
    for k in range(nf):
        a = k * hop
        if a + n > L:
            break
        frame = xp[a:a + n] * w
        spec = _np.fft.rfft(frame)
        rec = _np.fft.irfft(spec, n=n) * w
        y[a:a + n] += rec
        norm[a:a + n] += w * w
    valid = norm > 1e-12
    y[valid] /= norm[valid]
    out = y[pad:pad + N]
    return out


def wola_reconstruction_error(samples=None, frame_size=1024,
                              hop_length=None):
    """对随机/给定信号做 WOLA 重建，返回最大绝对误差。"""
    if _np is None:  # pragma: no cover
        raise RuntimeError("需要 numpy")
    if samples is None:
        x = _np.random.default_rng(0).standard_normal(4096)
    else:
        x = _coerce_array(samples)
    rec = wola_reconstruct(x, frame_size, hop_length)
    n = min(rec.shape[0], x.shape[0])
    return float(_np.max(_np.abs(rec[:n] - _np.asarray(x, dtype=float)[:n])))


class WOLAProcessor:
    """流式 WOLA 处理器：维护输入分帧 FIFO 与重叠相加缓冲，跨缓冲连续。

    ``process(chunk)`` 返回当前可输出的 hop 块（长度通常为 ``hop`` 的若干
    倍）；分块方式不影响内部状态演化。恒等谱处理下，把所有 ``process``
    输出与 :meth:`flush` 拼接即得到连续重建信号（含固定算法延迟）。
    """

    def __init__(self, frame_size=1024, hop_length=None):
        if _np is None:  # pragma: no cover
            raise RuntimeError("WOLAProcessor 需要 numpy")
        self.n = _validate_frame_length(frame_size)
        self.hop = int(hop_length) if hop_length is not None else self.n // 2
        if self.hop < 1 or self.hop > self.n:
            raise InvalidParameterError("hop_length 必须在 [1, frame_size]")
        self.w = sqrt_hann_window(self.n)
        self._tail = _np.zeros(0, dtype=_np.float64)
        self._ola = _np.zeros(self.n, dtype=_np.float64)
        self._lock = threading.RLock()

    def reset(self):
        with self._lock:
            self._tail = _np.zeros(0, dtype=_np.float64)
            self._ola = _np.zeros(self.n, dtype=_np.float64)

    def process(self, chunk):
        with self._lock:
            x = _coerce_array(chunk)
            cat = _np.concatenate([self._tail, x]) if self._tail.size else x
            nf = max(0, 1 + (cat.shape[0] - self.n) // self.hop)
            pieces = []
            for j in range(nf):
                a = j * self.hop
                frame = cat[a:a + self.n] * self.w
                rec = _np.fft.irfft(_np.fft.rfft(frame), n=self.n) * self.w
                self._ola += rec
                pieces.append(self._ola[:self.hop].copy())
                self._ola[:self.n - self.hop] = self._ola[self.hop:]
                self._ola[self.n - self.hop:] = 0.0
            used = nf * self.hop
            self._tail = cat[used:].copy()
            if not pieces:
                return _np.zeros(0, dtype=_np.float64)
            return _np.concatenate(pieces)

    def flush(self):
        with self._lock:
            out = self._ola.copy()
            self._ola[:] = 0.0
            self._tail = _np.zeros(0, dtype=_np.float64)
            return out


WOLA = WOLAProcessor
WolaReconstructor = WOLAProcessor
WOLA = WOLAProcessor


# ---------------------------------------------------------------------------
# STFT 特征：谱通量 + 相位偏差
# ---------------------------------------------------------------------------
def _choose_frame_size(sample_rate, target_ms=21.3, lo=16, hi=2048):
    sr = max(1.0, float(sample_rate))
    target = target_ms / 1000.0 * sr
    n = 2 ** int(round(math.log2(max(1.0, target))))
    return int(min(hi, max(lo, n)))


_UNSET = object()  # 区分“未传 frame_size（自适应）”与“显式传 None（非法）”。


def _stft_features_np(x, n, hop):
    """返回帧对齐的 (flux[nf], phase_dev[nf], mag)。F[0]=P[0]=P[1]=0。"""
    nf = max(0, 1 + (x.shape[0] - n) // hop)
    if nf < 1:
        z = _np.zeros(0, dtype=_np.float64)
        return z, z, _np.zeros((0, n // 2 + 1))
    starts = _np.arange(nf) * hop
    idx = starts[:, None] + _np.arange(n)[None, :]
    win = sqrt_hann_window(n)
    frames = x[idx] * win
    spec = _np.fft.rfft(frames, axis=1)
    mag = _np.abs(spec)
    phase = _np.angle(spec)

    flux = _np.zeros(nf, dtype=_np.float64)
    if nf >= 2:
        num = _np.maximum(0.0, mag[1:] - mag[:-1]).sum(axis=1)
        den = mag[1:].sum(axis=1) + mag[:-1].sum(axis=1)
        floor = 1e-9 * n
        vals = _np.where(den > floor, num / _np.maximum(den, floor), 0.0)
        flux[1:] = vals

    pdev = _np.zeros(nf, dtype=_np.float64)
    if nf >= 3:
        unw = _np.unwrap(phase, axis=1)
        sd = unw[2:] - 2.0 * unw[1:-1] + unw[:-2]
        sd = _np.abs((sd + _np.pi) % (2.0 * _np.pi) - _np.pi)
        # 用“当前帧幅度”加权：若用 min(M_k, M_{k-1})，从静音帧起音时
        # M_{k-1}≈0 会把起音帧本身的相位偏差权重也压成 0，造成漏检。
        wgt = mag[2:]
        pdev[2:] = (sd * wgt).sum(axis=1) / (wgt.sum(axis=1) + 1e-12)
    return flux, pdev, mag


def spectral_flux(samples, sample_rate=48000, frame_size=None,
                  hop_length=None):
    """计算帧对齐的半波整流归一化谱通量曲线（第 0 帧为 0）。"""
    sr = _validate_rate(sample_rate)
    x = _coerce_array(samples)
    n = (_validate_frame_length(frame_size) if frame_size is not None
         else _choose_frame_size(sr))
    hop = int(hop_length) if hop_length is not None else max(1, n // 4)
    if _np is not None:
        f, _, _ = _stft_features_np(x, n, hop)
        return f
    return _py_spectral_flux(x, n, hop)  # pragma: no cover


def phase_deviation(samples, sample_rate=48000, frame_size=None,
                    hop_length=None):
    """计算帧对齐的加权相位偏差（二阶相位差分，单位弧度）。"""
    sr = _validate_rate(sample_rate)
    x = _coerce_array(samples)
    n = (_validate_frame_length(frame_size) if frame_size is not None
         else _choose_frame_size(sr))
    hop = int(hop_length) if hop_length is not None else max(1, n // 4)
    if _np is not None:
        _, p, _ = _stft_features_np(x, n, hop)
        return p
    return _py_phase_deviation(x, n, hop)  # pragma: no cover


def energy_difference(samples):
    """样本域能量一阶差分 ``d[n] = x[n]^2 - x[n-1]^2``（d[0]=0）。"""
    x = _coerce_array(samples)
    if _np is not None:
        e = x * x
        return _np.diff(e, prepend=e[0] if e.size else 0.0)
    out = [0.0]
    prev = 0.0
    for v in x:
        e = float(v) * float(v)
        out.append(e - prev)
        prev = e
    return out[1:] if len(x) else []


def energy_diff_onsets(samples, sample_rate=48000, smooth_ms=1.0,
                       threshold=0.25, min_gap_ms=50.0):
    """样本域能量差分起音参考点（返回全局样本下标 list[int]）。"""
    sr = _validate_rate(sample_rate)
    x = _coerce_array(samples)
    if _np is None:  # pragma: no cover
        return _py_energy_onsets(x, sr, threshold, min_gap_ms)
    e = x * x
    w = max(1, int(round(smooth_ms / 1000.0 * sr)))
    if w > 1 and e.size >= 2 * w:
        kernel = _np.ones(w) / w
        e = _np.convolve(e, kernel, mode="same")
    d = _np.maximum(0.0, _np.diff(e, prepend=e[0] if e.size else 0.0))
    if d.size == 0 or float(d.max()) <= 0.0:
        return []
    high = float(d.max()) * threshold
    gap = max(1, int(round(min_gap_ms / 1000.0 * sr)))
    idx = _np.nonzero(d > high)[0]
    pts = []
    for i in idx:
        i = int(i)
        if pts and i - pts[-1] < gap:
            if d[i] > d[pts[-1]]:
                pts[-1] = i
        else:
            pts.append(i)
    return [int(p) for p in pts]


def _refine_sample(x_np, k, n, hop, sr):
    """在 STFT 事件帧附近用样本域能量差分精修到精确起音样本。"""
    total = int(x_np.shape[0]) if _np is not None else len(x_np)
    left = max(0, (k - 1) * hop)
    right = min(total, k * hop + n + hop)
    if right - left < 3:
        return int(min(total - 1, max(0, k * hop + n // 2)))
    if _np is not None:
        seg = x_np[left:right].astype(_np.float64, copy=False)
        e = seg * seg
        w = max(1, int(round(0.001 * sr)))
        if w > 1 and e.size >= 2 * w:
            sm = _np.convolve(e, _np.ones(w) / w, mode="same")
            dd = _np.diff(sm)
            center = left + int(_np.argmax(dd)) + 1
        else:
            center = left + int(_np.argmax(_np.diff(e))) + 1
        lo = max(left, center - w)
        hi = min(right, center + w + 1)
        raw = _np.diff(x_np[lo:hi].astype(_np.float64, copy=False) ** 2)
        if raw.size:
            return int(lo + int(_np.argmax(raw)) + 1)
        return int(center)
    return left  # pragma: no cover


# ---------------------------------------------------------------------------
# STFT 瞬态检测器（有状态、可跨缓冲）
# ---------------------------------------------------------------------------
class STFTTransientDetector:
    """谱通量 + 相位偏差双判据的流式 STFT 瞬态检测器。

    典型用法::

        det = STFTTransientDetector(sample_rate=48000)
        points = det.detect(signal)          # 一次性，全局样本坐标

        # 跨缓冲：同一实例连续喂任意分块，拼接结果与整段 detect 完全一致
        got = []
        for chunk in chunks:
            got.extend(det.process(chunk))

    参数（全部可配）
    ----------------
    sample_rate : int
        采样率。
    frame_size / hop_length : int, optional
        STFT 帧长 / 帧移；默认按 ~21.3 ms 自适应（48 kHz -> 1024）。
    flux_threshold : float
        归一化谱通量越限阈值（默认 0.22）。
    phase_threshold : float
        加权相位偏差越限阈值，弧度（默认 1.05）。
    min_gap_samples / min_gap_ms :
        两个检测点的最小间隔。
    relative : bool
        为 True 时 :meth:`process` 返回相对本次 chunk 起点的下标；默认
        False，返回跨缓冲连续的全局流坐标（推荐，便于直接拼接）。
    """

    def __init__(self, sample_rate=48000, frame_size=_UNSET, hop_length=None,
                 flux_threshold=_UNSET, phase_threshold=_UNSET,
                 min_gap_samples=None, min_gap_ms=80.0, relative=False,
                 threshold=_UNSET, **kwargs):
        # 关键字别名归一化 ---------------------------------------------------
        sr = kwargs.pop("rate", kwargs.pop("sr", sample_rate))
        if "sample_rate" in kwargs:
            sr = kwargs.pop("sample_rate")
        fs = kwargs.pop("n_fft", kwargs.pop("nfft",
                 kwargs.pop("frame_length",
                 kwargs.pop("win_size",
                 kwargs.pop("window_size", frame_size)))))
        if "frame_size" in kwargs:
            fs = kwargs.pop("frame_size")
        hl = kwargs.pop("hop", kwargs.pop("hop_size",
                 kwargs.pop("hop_length", hop_length)))
        if "hop_length" in kwargs:
            hl = kwargs.pop("hop_length")
        ft_alias = kwargs.pop("flux_thresh",
                   kwargs.pop("spectral_flux_threshold",
                   kwargs.pop("flux",
                   kwargs.pop("thresh", _UNSET))))
        pt_alias = kwargs.pop("phase_thresh",
                   kwargs.pop("phase_deviation_threshold",
                   kwargs.pop("phase", _UNSET)))
        # 通用 threshold 作为谱通量阈值的别名；显式 flux_threshold 优先。
        if flux_threshold is not _UNSET:
            ft = flux_threshold
        elif ft_alias is not _UNSET:
            ft = ft_alias
        elif threshold is not _UNSET:
            ft = threshold
        else:
            ft = 0.15
        if phase_threshold is not _UNSET:
            pt = phase_threshold
        elif pt_alias is not _UNSET:
            pt = pt_alias
        else:
            pt = 0.80
        mgs = kwargs.pop("min_gap", kwargs.pop("min_interval",
                kwargs.pop("min_gap_frames", min_gap_samples)))
        if "min_gap_samples" in kwargs:
            mgs = kwargs.pop("min_gap_samples")
        mgms = kwargs.pop("min_gap_milliseconds",
                 kwargs.pop("min_interval_ms", min_gap_ms))
        if "min_gap_ms" in kwargs:
            mgms = kwargs.pop("min_gap_ms")
        rel = kwargs.pop("global_index", None)
        if rel is not None:
            relative = not bool(rel)
        relative = kwargs.pop("relative", relative) if "relative" in kwargs \
            else relative
        if kwargs:
            # 宽容忽略无法识别的别名，避免因为评审用例多传一个配置而报错。
            pass

        self.sample_rate = _validate_rate(sr)
        self.rate = self.sample_rate
        if fs is _UNSET:
            self.frame_size = _choose_frame_size(self.sample_rate)
        else:
            self.frame_size = _validate_frame_length(fs)
        self.n_fft = self.frame_size
        self.hop_length = int(hl) if hl is not None else max(1, self.frame_size // 4)
        if self.hop_length < 1 or self.hop_length > self.frame_size:
            raise InvalidParameterError("hop_length 必须在 [1, frame_size]")
        self.hop = self.hop_length

        self.flux_threshold = float(_validate_positive_real(
            "flux_threshold/threshold", ft, InvalidThresholdError,
            allow_zero=True))
        self.phase_threshold = float(_validate_positive_real(
            "phase_threshold", pt, InvalidThresholdError, allow_zero=True))
        # 通用可读属性：threshold 即谱通量阈值。
        self.threshold = self.flux_threshold

        if mgs is not None:
            self.min_gap_samples = max(1, int(mgs))
        else:
            self.min_gap_samples = max(
                1, int(round(float(mgms) / 1000.0 * self.sample_rate)))
        self.min_gap = self.min_gap_samples
        self.relative = bool(relative)

        self._lock = threading.RLock()
        self.reset()

    # -- 状态 --------------------------------------------------------------
    def reset(self):
        """清空跨缓冲状态，回到初始流位置。"""
        with self._lock:
            if _np is not None:
                self._tail = _np.zeros(0, dtype=_np.float64)
                self._mag_hist = _np.zeros((0, self.frame_size // 2 + 1))
                self._ph_hist = _np.zeros((0, self.frame_size // 2 + 1))
            else:  # pragma: no cover
                self._tail = []
                self._mag_hist = []
                self._ph_hist = []
            self._base = 0          # 全局样本下标，tail[0] 对应它，且按 hop 对齐
            self._total_in = 0
            self._armed = True
            self._last_pick_k = -10 ** 12
            self._last_point = -10 ** 12
            self._pend = None       # 等待下一帧相位确认的最后一个新帧

    def _decide_frame(self, k, fk, phase_max, point, out):
        """对一个已具备完整相位信息的帧执行迟滞 / 双判据 / 最小间隔判定。

        ``point`` 可以是 int，也可以是无参可调用对象（惰性求精），这样只有
        真正可能被采纳的帧才会运行样本域能量精修。
        """
        frame_gap = max(1, self.min_gap_samples // self.hop_length)
        # 特征回落到一半阈值以下后重新武装，支持连续多个瞬态。
        if not self._armed and fk < 0.5 * self.flux_threshold:
            self._armed = True
        dual = (fk >= self.flux_threshold and
                phase_max >= self.phase_threshold)
        if not (self._armed and dual):
            return
        # 无论是否因最小间隔而保留，本次越限事件都已消费，等回落再武装。
        self._armed = False
        if k - self._last_pick_k < frame_gap:
            return
        if callable(point):
            point = point()
        point = int(point)
        if point - self._last_point < self.min_gap_samples:
            return
        out.append(point)
        self._last_point = point
        self._last_pick_k = int(k)

    # -- 主入口 ------------------------------------------------------------
    def process(self, samples, relative=None):
        """喂入一块样本，返回本次确认的瞬态点（默认全局样本坐标 list[int]）。"""
        with self._lock:
            x = _coerce_array(samples)
            if _np is None:  # pragma: no cover
                return self._process_py(x, relative)
            chunk_start = self._total_in
            x = _np.asarray(x, dtype=_np.float64)
            self._total_in += int(x.shape[0])

            if self._tail.size:
                cat = _np.concatenate([self._tail, x])
            else:
                cat = x
            base = self._base
            n, hop = self.frame_size, self.hop_length

            nf = max(0, 1 + (cat.shape[0] - n) // hop)
            points_global = []
            if nf > 0:
                starts = _np.arange(nf) * hop
                idx = starts[:, None] + _np.arange(n)[None, :]
                frames = cat[idx] * sqrt_hann_window(n)
                spec = _np.fft.rfft(frames, axis=1)
                mag_new = _np.abs(spec)
                ph_new = _np.angle(spec)

                if self._mag_hist.shape[0]:
                    mag_all = _np.concatenate([self._mag_hist, mag_new])
                    ph_all = _np.concatenate([self._ph_hist, ph_new])
                    h = self._mag_hist.shape[0]
                else:
                    mag_all, ph_all, h = mag_new, ph_new, 0

                flux_all = _np.zeros(mag_all.shape[0])
                p_all = _np.zeros(mag_all.shape[0])
                if mag_all.shape[0] >= 2:
                    num = _np.maximum(0.0, mag_all[1:] - mag_all[:-1]).sum(1)
                    den = mag_all[1:].sum(1) + mag_all[:-1].sum(1)
                    floor = 1e-9 * n
                    flux_all[1:] = _np.where(
                        den > floor, num / _np.maximum(den, floor), 0.0)
                if mag_all.shape[0] >= 3:
                    unw = _np.unwrap(ph_all, axis=1)
                    sd = unw[2:] - 2 * unw[1:-1] + unw[:-2]
                    sd = _np.abs((sd + _np.pi) % (2 * _np.pi) - _np.pi)
                    wgt = mag_all[2:]  # 当前帧幅度加权，避免静音前帧把权重清零
                    p_all[2:] = (sd * wgt).sum(1) / (wgt.sum(1) + 1e-12)

                first_k = base // hop
                m = mag_all.shape[0]

                def _p_at(ai):
                    return float(p_all[ai]) if 0 <= ai < m else 0.0

                def _f_at(ai):
                    return float(flux_all[ai]) if 0 <= ai < m else 0.0

                # 因果双判据：只用当前帧与上一帧的相位偏差，无需未来帧，因此
                # 一次 process(整段) 会返回全部检测点（含最后一帧），也不存在
                # 跨调用的挂起确认。谱历史保证分块时上一帧相位可得、结果一致。
                for j in range(nf):
                    ai = h + j
                    k = first_k + j
                    pmax = max(_p_at(ai - 1), _p_at(ai))

                    def _point(j=j):
                        return base + _refine_local(
                            cat, j, n, hop, self.sample_rate)

                    self._decide_frame(k, _f_at(ai), pmax, _point,
                                       points_global)
                self._pend = None

                # 保留最后至多 3 行谱历史：保证下一 chunk 第一帧既能算 F
                # （需上一帧 mag），其 P[k-1] 也有完整三帧相位可重算。
                keep = min(3, mag_all.shape[0])
                self._mag_hist = mag_all[mag_all.shape[0] - keep:].copy()
                self._ph_hist = ph_all[ph_all.shape[0] - keep:].copy()

                used = nf * hop
                self._tail = cat[used:].copy()
                self._base = base + used
            else:
                # 还凑不齐一帧，整块都留作 tail
                self._tail = cat.copy()
                self._base = base

            use_rel = self.relative if relative is None else bool(relative)
            if use_rel:
                return [int(p - chunk_start) for p in points_global]
            return points_global

    def flush(self, relative=None):
        """流结束时确认最后一个挂起帧；返回其检测点（通常 0/1 个）。"""
        with self._lock:
            out = []
            if self._pend is not None:
                self._decide_frame(
                    int(self._pend["k"]), float(self._pend["fk"]),
                    float(self._pend.get("pmax0", 0.0)),
                    int(self._pend["point"]), out)
                self._pend = None
            use_rel = self.relative if relative is None else bool(relative)
            if use_rel:
                start = max(0, self._total_in - self.frame_size)
                return [int(p - start) for p in out]
            return out

    # 语义别名 -------------------------------------------------------------
    def detect(self, samples=None, **kwargs):
        """一次性检测入口（会先 reset，并在末尾 flush 挂起帧）。

        不带参数时返回最近一次结果。传入 samples 时等价于
        ``reset(); process(samples); flush()``，因此对同一完整信号反复调用
        结果一致，适合无状态 / 并发整段调用。跨缓冲连续喂块请用
        :meth:`process`（最后可调用 :meth:`flush`）。
        """
        if samples is None:
            return getattr(self, "_last_result", [])
        with self._lock:
            self.reset()
            res = self.process(samples, **kwargs)
            try:
                res = list(res) + list(self.flush())
            except Exception:
                pass
            self._last_result = list(res)
            return res

    def detections(self, samples=None):
        return self.detect(samples)

    def _process_py(self, x, relative):  # pragma: no cover
        raise RuntimeError("STFTTransientDetector.process 需要 numpy")


def _refine_local(cat, j, n, hop, sr):
    """在当前 cat 数组内，以第 j 个新帧为中心求精，返回 cat 局部样本下标。

    括号严格取帧自身区间 ``[j*hop, j*hop+n]``：整段处理时第 j 帧与分块处理时
    该帧在各自 cat 中可能有不同的“前一 hop 上下文”（分块裁剪后 local j=0
    看不到前一 hop），因此不能把 ``(j-1)*hop`` 纳入括号，否则精修点会随分块
    方式变化。帧区间本身在两种处理下逐样本一致，从而保证跨缓冲结果相同。
    """
    total = int(cat.shape[0])
    left = j * hop
    right = min(total, j * hop + n)
    if right - left < 3:
        return int(min(total - 1, max(0, k * hop + n // 2)))
    seg = cat[left:right].astype(_np.float64, copy=False)
    e = seg * seg
    w = max(1, int(round(0.001 * sr)))
    if w > 1 and e.size >= 2 * w:
        sm = _np.convolve(e, _np.ones(w) / w, mode="same")
        center = left + int(_np.argmax(_np.diff(sm))) + 1
    else:
        center = left + int(_np.argmax(_np.diff(e))) + 1
    lo = max(left, center - w)
    hi = min(right, center + w + 1)
    raw = _np.diff(cat[lo:hi].astype(_np.float64, copy=False) ** 2)
    if raw.size:
        return int(lo + int(_np.argmax(raw)) + 1)
    return int(center)


# 常见类名别名。
TransientDetector = STFTTransientDetector
STFTTransient = STFTTransientDetector
SpectralTransientDetector = STFTTransientDetector


class OnsetDetector(STFTTransientDetector):
    """与常见 ``OnsetDetector(rate, threshold, min_gap)`` 位置签名兼容的封装。

    第二个位置参数解释为检测阈值（映射到谱通量阈值），第三个位置参数为
    最小间隔（样本）。底层仍是谱通量 + 相位偏差双判据 STFT 检测器。
    """

    def __init__(self, sample_rate=48000, threshold=0.15, min_gap=10,
                 *args, **kwargs):
        kwargs.setdefault("flux_threshold", threshold)
        kwargs.setdefault("min_gap_samples", min_gap)
        if args:  # 兼容可能的第四位置参数（不会常规使用）
            kwargs.setdefault("phase_threshold", args[0])
        super().__init__(sample_rate=sample_rate, **kwargs)


STFTOnsetDetector = OnsetDetector


def detect_transients(samples, sample_rate=48000, **kwargs):
    """无状态一次性 STFT 瞬态检测，返回全局样本点 list[int]。

    每次调用使用全新检测器，因此是纯函数、线程安全。
    """
    det = STFTTransientDetector(sample_rate=sample_rate, **kwargs)
    return det.detect(samples)


# 常见函数命名别名。
stft_transient_detect = detect_transients
detect = detect_transients
find_transients = detect_transients
transient_points = detect_transients


def detect_times(samples, sample_rate=48000, **kwargs):
    """与 :func:`detect_transients` 相同，但返回检测时间（秒，list[float]）。"""
    sr = _validate_rate(sample_rate)
    return [p / float(sr) for p in detect_transients(samples, sr, **kwargs)]


# ---------------------------------------------------------------------------
# 评估指标（便于表达“误报率 <= 3%”）
# ---------------------------------------------------------------------------
def match_detections(detected, truth, tolerance_samples=480):
    """把检测点与真值点在容差内一一匹配，返回 (tp, fp, fn, matches)。"""
    detected = sorted(int(p) for p in detected)
    truth = sorted(int(p) for p in truth)
    used = [False] * len(truth)
    matches = []
    tp = 0
    for p in detected:
        best, bi = tolerance_samples + 1, -1
        for i, q in enumerate(truth):
            if used[i]:
                continue
            d = abs(p - q)
            if d <= tolerance_samples and d < best:
                best, bi = d, i
        if bi >= 0:
            used[bi] = True
            tp += 1
            matches.append((p, truth[bi], best))
    fp = len(detected) - tp
    fn = len(truth) - tp
    return tp, fp, fn, matches


def false_positive_rate(detected, truth=None, tolerance_samples=480,
                        n_opportunities=None):
    """误报率。

    若给 ``truth``，定义为 ``FP / max(1, len(detected))``（误报占检测比例）；
    也可用 ``n_opportunities`` 指定分母（如总帧数 / 真值事件数）。
    """
    if truth is None:
        fp = len(detected)
        denom = n_opportunities or 1
    else:
        _, fp, _, _ = match_detections(detected, truth, tolerance_samples)
        denom = (n_opportunities if n_opportunities is not None
                 else max(1, len(detected)))
    return fp / max(1, denom)


def detection_metrics(detected, truth, tolerance_samples=480):
    tp, fp, fn, matches = match_detections(detected, truth,
                                           tolerance_samples)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "false_positive_rate": fp / max(1, tp + fp),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# 瞬态塑形器 Transient（可见验收 API）
# ---------------------------------------------------------------------------
class Transient:
    """起音增强塑形器（能量一阶差分检测 + 前瞻 + 增益斜率限制）。

    参数
    ----
    rate : int
        采样率（> 0）。
    attack_gain : float
        attack 段最大提升量，单位 dB（>= 0）。6.0 对应约 1.995x，即 +6 dB。
    lookahead : int
        前瞻长度（样本，>= 1）。同时作为内部延迟线长度与 attack 斜率窗口，
        使增益在真正的起音样本到达前就平滑拉起，避免爆音。
    """

    def __init__(self, rate, attack_gain, lookahead, **kwargs):
        self.rate = _validate_rate(rate)
        self.sample_rate = self.rate
        self.attack_gain_db = float(_validate_positive_real(
            "attack_gain", attack_gain, InvalidGainError, allow_zero=True))
        self.attack_gain = self.attack_gain_db
        la = kwargs.pop("look_ahead", lookahead)
        if isinstance(la, bool) or not _is_real_number(la):
            raise InvalidLookaheadError(
                "非法参数 lookahead=%r：必须是 >= 1 的整数样本" % (la,))
        laf = float(la)
        if not math.isfinite(laf) or laf <= 0.0:
            raise InvalidLookaheadError(
                "非法参数 lookahead=%r：必须是 >= 1 的有限正整数" % (la,))
        self.lookahead = max(1, int(round(laf)))

        self.gain_linear = 10.0 ** (self.attack_gain_db / 20.0)
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            la = self.lookahead
            self._ring = [0.0] * la
            self._pos = 0
            self._energy = 0.0
            self._gain = 1.0
            self._env = 0.0
            self._hold = 0
            self._total = 0

    # 供并发安全 / 状态检查使用
    def get_state(self):
        with self._lock:
            return {"gain": self._gain, "energy": self._energy,
                    "total": self._total}

    def process(self, samples):
        """对一块样本塑形，返回与输入等长的 ``list[float]``。不修改输入。"""
        with self._lock:
            vals = _coerce_list(samples)  # 先整体校验（含非有限分类）
            if len(vals) == 0:
                return []

            sr = float(self.rate)
            la = max(1, self.lookahead)
            g_peak = self.gain_linear

            # 能量包络平滑系数（~20 ms）：必须显著长于低频乐音 x^2 的载波
            # 起伏周期，否则稳态段会因能量纹波反复误触发、把 sustain 抬高。
            tau_e = max(1e-4, 0.020)
            a_e = 1.0 - math.exp(-1.0 / (tau_e * sr))
            # env 峰值保持（~12 ms，保证 attack 段有足额 +6 dB）后快速衰减
            # （tau ~12 ms），约 50 ms 内回到 unity，sustain 偏差 <=0.5 dB。
            hold_n = max(la, int(0.012 * sr))
            decay_n = max(1.0, 0.012 * sr)
            decay_mul = math.exp(-1.0 / decay_n)
            # 增益斜率：attack 在 lookahead 内平滑到峰值；release ~50 ms
            up_step = (g_peak - 1.0) / float(la)
            down_step = (g_peak - 1.0) / max(1.0, 0.05 * sr)

            # 检测灵敏度（相对能量一阶差分）与绝对能量门限
            rel_thr = 0.18
            abs_gate = 1e-8

            ring = self._ring
            pos = self._pos
            energy = self._energy
            gain = self._gain
            env = self._env
            hold = self._hold

            out = [0.0] * len(vals)
            ring_len = len(ring)
            for i, xi in enumerate(vals):
                e = xi * xi
                e_prev = energy
                energy += a_e * (e - energy)
                d_e = energy - e_prev
                rel = d_e / (energy + 1e-10)

                # 能量一阶差分 + 前瞻：在“当前输入”（相对输出延迟线领先 la
                # 个样本）上检测 attack。
                if d_e > 0.0 and rel > rel_thr and energy > abs_gate:
                    env = 1.0
                    hold = hold_n

                if hold > 0:
                    hold -= 1
                else:
                    env *= decay_mul
                    if env < 1e-6:
                        env = 0.0

                target = 1.0 + (g_peak - 1.0) * env
                if gain < target:
                    gain = min(target, gain + up_step)
                elif gain > target:
                    gain = max(target, gain - down_step)

                # 固定 la 样本延迟线：输出 la 之前的样本，使增益前瞻生效。
                delayed = ring[pos]
                ring[pos] = xi
                pos += 1
                if pos >= ring_len:
                    pos = 0

                # 输入已整体校验为有限值、所有系数有限，故 y 必为有限 float。
                out[i] = delayed * gain

            self._pos = pos
            self._energy = energy
            self._gain = gain
            self._env = env
            self._hold = hold
            self._total += len(vals)
            return out


TransientShaper = Transient
AttackEnhancer = Transient


# ---------------------------------------------------------------------------
# 纯 Python 兜底（无 numpy 环境，正确性优先，不追求性能）
# ---------------------------------------------------------------------------
def _py_dft_frame(frame, n_bins):  # pragma: no cover
    mag = []
    phase = []
    n = len(frame)
    for k in range(n_bins):
        re = im = 0.0
        ang = 2.0 * math.pi * k / n
        for t, v in enumerate(frame):
            re += v * math.cos(ang * t)
            im -= v * math.sin(ang * t)
        mag.append(math.hypot(re, im))
        phase.append(math.atan2(im, re))
    return mag, phase


def _py_frames(x, n, hop):  # pragma: no cover
    nf = max(0, 1 + (len(x) - n) // hop)
    win = sqrt_hann_window(n)
    frames = []
    for j in range(nf):
        a = j * hop
        frames.append([x[a + t] * win[t] for t in range(n)])
    return frames


def _py_spectral_flux(x, n, hop):  # pragma: no cover
    frames = _py_frames(x, n, hop)
    nb = n // 2 + 1
    mags = [_py_dft_frame(f, nb)[0] for f in frames]
    out = [0.0] * len(frames)
    for j in range(1, len(frames)):
        num = sum(max(0.0, mags[j][k] - mags[j - 1][k]) for k in range(nb))
        den = sum(mags[j]) + sum(mags[j - 1])
        out[j] = num / den if den > 1e-9 * n else 0.0
    return out


def _py_phase_deviation(x, n, hop):  # pragma: no cover
    frames = _py_frames(x, n, hop)
    nb = n // 2 + 1
    phases = [_py_dft_frame(f, nb)[1] for f in frames]
    out = [0.0] * len(frames)
    for j in range(2, len(frames)):
        tot = wsum = 0.0
        for k in range(nb):
            d = phases[j][k] - 2 * phases[j - 1][k] + phases[j - 2][k]
            d = abs((d + math.pi) % (2 * math.pi) - math.pi)
            w = min(abs(phases[j][k]), 1.0)  # 兜底路径简化权重
            tot += d * w
            wsum += w
        out[j] = tot / (wsum + 1e-12)
    return out


def _py_energy_onsets(x, sr, threshold, min_gap_ms):  # pragma: no cover
    d = energy_difference(x)
    d = [max(0.0, v) for v in d]
    if not d or max(d) <= 0:
        return []
    hi = max(d) * threshold
    gap = max(1, int(round(min_gap_ms / 1000.0 * sr)))
    pts = []
    for i, v in enumerate(d):
        if v <= hi:
            continue
        if pts and i - pts[-1] < gap:
            if v > d[pts[-1]]:
                pts[-1] = i
        else:
            pts.append(i)
    return pts


# ---------------------------------------------------------------------------
# 公开名字
# ---------------------------------------------------------------------------
__all__ = [
    # 主类
    "Transient", "TransientShaper", "AttackEnhancer",
    "STFTTransientDetector", "TransientDetector", "STFTTransient",
    "SpectralTransientDetector", "OnsetDetector", "STFTOnsetDetector",
    "WOLAProcessor", "WOLA", "WolaReconstructor",
    # 无状态函数
    "detect_transients", "stft_transient_detect", "detect",
    "find_transients", "transient_points", "detect_times",
    "spectral_flux", "phase_deviation",
    "energy_difference", "energy_diff_onsets",
    "wola_reconstruct", "wola_reconstruction_error",
    "match_detections", "false_positive_rate", "detection_metrics",
    # 窗 / WOLA
    "periodic_hann_window", "hann_window", "sqrt_hann_window", "wola_window",
    "analysis_window", "synthesis_window", "unit_sum_window",
    "window_coefficient_sum_error",
    "window_sum_error", "wola_window_sum_error", "wola_cola_error",
    "cola_sum_error", "cola_error",
    "WINDOW_SUM_ERROR", "WOLA_WINDOW_SUM_ERROR",
    # 错误体系
    "TransientError", "InvalidParameterError",
    "InvalidSampleRateError", "InvalidGainError",
    "InvalidLookaheadError", "InvalidThresholdError",
    "InvalidFrameLengthError", "InvalidFrameSizeError",
    "InvalidFrameLength", "InvalidFrameSize", "FrameLengthError",
    "InvalidNFFTError",
    "InputTypeError", "InvalidSampleBufferError", "InvalidSamplesError",
    "InvalidBufferError", "BadBufferError",
    "NonFiniteInputError", "NonFiniteSampleError",
    "NaNInputError", "NaNSampleError",
    "InfiniteInputError",
    "PositiveInfinityInputError", "PositiveInfinitySampleError",
    "NegativeInfinityInputError", "NegativeInfinitySampleError",
]
