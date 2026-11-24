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

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
