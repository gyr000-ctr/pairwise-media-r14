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

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
