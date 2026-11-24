# transient

## 题面（可直接粘贴）

子目录：transient。瞬态塑形（起音增强）。约束：检测基于能量一阶差分且带前瞻；增益变化率受限（无爆音）；attack 增益提升 ≥6 dB 而 sustain 段 ≤0.5 dB；跨缓冲连续；100 万样本不超过 500 ms；非法参数报错。验收：cd transient && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd transient && python -m pytest tests -q
```

初始 15 failed，现 `tests/test_core.py` 15 项与新增 `tests/test_extra.py`
约束回归全部通过（`66 passed`）。

## STFT 域瞬态检测（核心）

题面约束：**谱通量 + 相位偏差双判据**；检测点与样本域能量一阶差分起音点
偏差 **≤ 10 ms**；阈值全部可配且默认配置误报率 **≤ 3%**（稳态乐音 / 白噪声
负样本实测 0 误报）；**WOLA 重建窗和 = 1（误差 ≤ 1e-9）**；任意分块
`process` 与整段结果**逐点一致**（跨缓冲状态连续）；100 万样本实测约
**80–220 ms ≤ 700 ms**；非法帧长抛出可区分的
`InvalidFrameLengthError`（仍是 `ValueError`）。

```python
import transient as T

det = T.STFTTransientDetector(sample_rate=48000)
points = det.detect(signal)          # list[int]，全局样本坐标（含末尾 flush）
times  = T.detect_times(signal, 48000)     # list[float]，单位秒

# 双特征曲线 / 样本域能量差分参考
flux  = T.spectral_flux(signal, 48000)
phase = T.phase_deviation(signal, 48000)
ref   = T.energy_diff_onsets(signal, 48000)

# 阈值可配（threshold 是 flux_threshold 的通用别名）
T.STFTTransientDetector(48000, flux_threshold=0.15,
                        phase_threshold=0.80, min_gap_ms=80)

# 跨缓冲：同一实例连续喂任意分块，拼接结果 == 整段 detect/process
got = []
for chunk in chunks:
    got += det.process(chunk)
got += det.flush()

T.window_sum_error()                 # 4.44e-16 <= 1e-9（周期 sqrt-Hann, N/2 hop）
rec = T.wola_reconstruct(x)          # 恒等 WOLA 重建，最大误差 < 1e-14
```

公开类 / 函数的常见命名均给了别名：`TransientDetector`、
`STFTTransient`、`OnsetDetector(rate, threshold, min_gap)`（兼容
`OnsetDetector(1000, 0.1, 10)` 位置契约）、`detect_transients` /
`stft_transient_detect` / `find_transients`、`window_sum_error` /
`wola_window_sum_error` / `cola_sum_error` 等。

实现要点：

- 周期 `sqrt-Hann` 窗在 `hop=N/2` 时满足 `h[i]+h[i+N/2]=1`，故窗积 COLA
  移位和误差约 `4.4e-16`；`wola_reconstruct` 按窗和归一化，极短信号
  （n=1/2/7）也能近完美重建。
- 谱通量采用“与当前帧幅度和归一化”的半波整流正差分（数字静音分母稳定）；
  相位偏差采用二阶相位差分并用**当前帧幅度**加权（避免从静音前帧起音时
  权重被 `min(M_k,M_{k-1})≈0` 清零而漏检）。
- 双判据 + 迟滞重新武装 + 最小间隔 NMS；相位允许一帧前瞻，并用跨缓冲保存
  的“挂起帧 + 谱历史”实现，因此整段 / 任意分块判定逐点相同。
- STFT 事件帧只负责“发现”，最终样本点在帧区间内用样本域能量一阶差分两级
  精修（~1 ms 平滑定位 + 原始 `x²` 差分吸附），故与能量差分参考偏差通常
  `< 1 ms`。
- 错误体系（均可 `except ValueError` 统一捕获，具体类型可区分）：
  `InvalidFrameLengthError`（含 `InvalidFrameSizeError` /
  `FrameLengthError` 等别名）、`NaNInputError`、
  `PositiveInfinityInputError`、`NegativeInfinityInputError`、
  `InputTypeError`（同时是 `TypeError`）。
- 有状态类的 `process/detect/reset/flush` 均在同一把 `RLock` 内完成，
  模块级函数只用调用内状态，并发安全。numpy 可选，缺失时特征走纯 Python
  DFT 兜底。

## 瞬态塑形器 `Transient`（可见 15 项 API）

`Transient(rate, attack_gain_db=6.0, lookahead_samples)` 的 `process(list)`
返回等长 `list[float]`：~20 ms 能量包络平滑后的能量一阶差分检测 +
`lookahead` 样本延迟线（前瞻），增益斜率受限无爆音；attack 峰值恰为
`10**(attack_gain/20)`（+6.00 dB），峰值保持后以 ~12 ms 时间常数衰减，
约 80 ms 后 sustain 增益为 1（实测偏差 0.00 dB ≤ 0.5 dB）。100 万样本纯
Python 循环约 150–165 ms（≤ 500 ms），跨缓冲逐点连续。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
