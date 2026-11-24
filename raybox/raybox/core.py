"""raybox 核心 API（初始未实现）。"""


class RayBox:
    def intersect(self, origin, direction, box):
        """返回 tmin/tmax 或 None"""
        raise NotImplementedError

    def hit(self, origin, direction, box):
        """是否命中"""
        raise NotImplementedError
