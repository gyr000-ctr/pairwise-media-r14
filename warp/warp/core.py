"""warp 核心 API（初始未实现）。"""


class MeshWarp:
    def warp(self, image, grid, out_size):
        """变形"""
        raise NotImplementedError

    def sample(self, image, x, y):
        """采样"""
        raise NotImplementedError
