# raybox

## 题面（可直接粘贴）

子目录：raybox。射线-AABB 求交（数值稳健）。约束：与逐面求交对照（≤1e-9）；射线起点在盒内返回 0；平行轴正确；除零与 NaN 方向可区分报错；1 万次不超过 200 ms；并发安全。验收：cd raybox && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd raybox && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
