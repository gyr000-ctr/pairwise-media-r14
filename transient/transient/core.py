"""transient 核心 API（初始未实现）。"""


class Transient:
    def __init__(self, rate, attack_gain, lookahead):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """处理"""
        raise NotImplementedError
