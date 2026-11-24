"""timers 核心 API（初始未实现）。"""


class TimerWheel:
    def __init__(self, slots):
        """构造"""
        raise NotImplementedError

    def add(self, delay, name):
        """添加"""
        raise NotImplementedError

    def tick(self, dt):
        """推进并返回到期项"""
        raise NotImplementedError

    def cancel(self, name):
        """取消"""
        raise NotImplementedError
