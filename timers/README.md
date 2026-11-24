# timers

## 题面（可直接粘贴）

子目录：timers。时间轮定时器。约束：到期顺序稳定（同刻度按插入序）；取消后不再触发；回绕（大 dt）不漏触发；合法槽位与刻度校验；10 万定时器不超过 300 ms；并发安全。验收：cd timers && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd timers && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
