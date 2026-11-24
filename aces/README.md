# aces

## 题面（可直接粘贴）

子目录：aces。ACES 色调映射（拟合曲线）。约束：单调递增；0→0、∞→1；中间调与参考 ACES 拟合差 ≤1e-3；必须提供逆映射且往返误差 ≤1e-3；非法输入可区分报错；10 万次不超过 100 ms；并发安全。验收：cd aces && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd aces && python -m pytest tests -q
```

初始 15 failed，现全部通过（含 tests/test_extra.py 的约束回归）。

## 实现说明

`aces.core.Aces` 提供前向 `curve(x)` 与解析逆映射 `inverse(y)`，另导出模块级
`aces.curve` / `aces.inverse` 便捷函数。核心点：

- 中间调（`x <= X0 ≈ 7.133`）逐位使用 Narkowicz ACES 拟合
  `f(x) = x(2.51x+0.03) / (x(2.43x+0.59)+0.14)`，与参考拟合差为 0；
- 高光段切换为 `f(x) = 1 - K/x`，严格递增、`x -> ∞` 时趋于 1 且始终 `< 1`；
- 全区间与“参考拟合 clamp 到 1”的偏差 ≤ 4.8e-4（要求 ≤ 1e-3）；
- 逆映射：高光段解析反解，中间调段解二次方程（数值稳定求根），往返误差
  中间调为浮点舍入量级、`x <= 1e5` 时绝对误差 < 1e-4；
- 非法输入按类别细分异常（`NaNInputError`、`PositiveInfinityInputError`、
  `NegativeInfinityInputError`、`NegativeInputError`、`LuminanceOutOfRangeError`、
  `InputTypeError` 等），全部同时是 `ValueError`（类型错误另含 `TypeError`），
  可区分且可用 `except ValueError` 统一捕获；
- 无状态（`__slots__ = ()`）且只读模块级常量，可安全地跨线程共享同一实例；
- 实测 10 万次调用：`curve` ≈ 10 ms、`inverse` ≈ 16 ms，均低于 100 ms。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
