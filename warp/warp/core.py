"""warp 核心 API：基于控制点网格的双线性变形。

设计要点：

- ``MeshWarp`` 不保存任何可变实例状态，所有工作数据均为调用内的局部变量，
  因此天然可重入、并发安全（多线程共享同一实例或同一输入均无副作用）。
- 控制点处取数走 floor 下标、权重为精确的 0/1，保证控制点精确（误差仅浮点
  运算本身的舍入，远小于 1e-9）。
- 网格单元内部使用标准双线性插值；输出坐标先经逆映射得到单元内 (u, v)，
  再在图像上做双线性采样，整体 C0 连续。
- 越界坐标统一 clamp 到边界。
- 网格错误、尺寸错误、图像错误使用可区分的异常类型（均为 ``ValueError``
  子类），非有限输入统一归类为 ``ValueError``。
"""

import math
from bisect import bisect_right

__all__ = [
    "MeshWarp",
    "MeshWarpError",
    "GridError",
    "SizeError",
    "ImageError",
]


class MeshWarpError(ValueError):
    """MeshWarp 参数错误基类（仍是 ValueError）。"""


class GridError(MeshWarpError):
    """控制网格非法（空网格、结构不完整、全部退化等）。"""


class SizeError(MeshWarpError):
    """输出尺寸非法。"""


class ImageError(MeshWarpError):
    """输入图像非法（空、异形行、非有限像素等）。"""


def _validate_image(image):
    """校验图像，返回 (扁平 float 数组, 高, 宽)。"""
    if not isinstance(image, (list, tuple)) or len(image) == 0:
        raise ImageError("image must be a non-empty sequence of rows")
    height = len(image)
    width = None
    flat = []
    for row in image:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ImageError("image rows must be non-empty sequences")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ImageError("image rows must all have the same length")
        for value in row:
            if not isinstance(value, (int, float)):
                raise TypeError("image pixels must be numbers")
            if not math.isfinite(value):
                raise ImageError("image pixels must be finite")
            flat.append(float(value))
    return flat, height, width


def _validate_size(out_size):
    """校验输出尺寸 (高, 宽)，返回正整数元组。"""
    if not isinstance(out_size, (list, tuple)) or len(out_size) != 2:
        raise TypeError("out_size must be a (height, width) pair")
    dims = []
    for dim in out_size:
        if isinstance(dim, bool) or not isinstance(dim, int):
            if isinstance(dim, float) and not math.isfinite(dim):
                raise SizeError("out_size dimensions must be finite")
            raise TypeError("out_size dimensions must be integers")
        if dim <= 0:
            raise SizeError("out_size dimensions must be positive integers")
        dims.append(dim)
    return dims[0], dims[1]


def _validate_grid(grid):
    """校验控制点网格。

    网格为扁平单元序列，每个单元是 ((x0, y0), (x1, y1)) 形式的轴对齐矩形
    （p0/p1 为对角点）。有效单元必须拼成一个完整、无缝、无重叠的矩形网格；
    零宽度或零高度的退化单元测度为零、永不参与采样，容忍并忽略（测试夹具
    即包含此类单元）。

    返回 (cells, nx, ny, xs, ys)，其中 cells[j][i] 为 (x0, y0, x1, y1)。
    """
    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        raise GridError("grid must be a non-empty sequence of cells")

    valid_cells = []
    x_values = set()
    y_values = set()
    for cell in grid:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise GridError("each grid cell must contain exactly 2 points")
        p0, p1 = cell
        if not (isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple))) or (
            len(p0) != 2 or len(p1) != 2
        ):
            raise GridError("each grid point must be an (x, y) pair")
        x0, y0 = p0
        x1, y1 = p1
        for coord in (x0, y0, x1, y1):
            if not isinstance(coord, (int, float)):
                raise TypeError("grid coordinates must be numbers")
            if not math.isfinite(coord):
                raise GridError("grid coordinates must be finite")
        if x1 < x0 or y1 < y0:
            raise GridError(
                "grid cell corners must be ordered (x1 >= x0, y1 >= y0)"
            )
        # 退化单元（零宽或零高）不可能包含任何采样点，直接忽略。
        if x1 == x0 or y1 == y0:
            continue
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        valid_cells.append((x0, y0, x1, y1))
        x_values.add(x0)
        x_values.add(x1)
        y_values.add(y0)
        y_values.add(y1)

    if not valid_cells:
        raise GridError("grid must contain at least one non-degenerate cell")

    xs = sorted(x_values)
    ys = sorted(y_values)
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: j for j, y in enumerate(ys)}
    nx = len(xs) - 1
    ny = len(ys) - 1

    cells = [[None] * nx for _ in range(ny)]
    for x0, y0, x1, y1 in valid_cells:
        i = x_index[x0]
        j = y_index[y0]
        if x_index[x1] != i + 1 or y_index[y1] != j + 1:
            raise GridError(
                "grid cells must align into a regular rectangular mesh"
            )
        if not (0 <= i < nx and 0 <= j < ny) or cells[j][i] is not None:
            raise GridError("grid cells overlap or leave gaps")
        cells[j][i] = (x0, y0, x1, y1)

    if any(cell is None for row in cells for cell in row):
        raise GridError("grid does not cover a complete rectangular mesh")
    return cells, nx, ny, xs, ys


def _bilinear_sample(flat, width, height, gx, gy):
    """在归一化坐标 [0,1]^2 上做双线性采样，调用方负责 clamp 与有限性。"""
    gxf = gx * (width - 1)
    gyf = gy * (height - 1)
    ix0 = int(gxf)
    iy0 = int(gyf)
    if ix0 >= width - 1:
        ix0 = ix1 = width - 1
        tx = 0.0
    else:
        ix1 = ix0 + 1
        tx = gxf - ix0
    if iy0 >= height - 1:
        iy0 = iy1 = height - 1
        ty = 0.0
    else:
        iy1 = iy0 + 1
        ty = gyf - iy0
    base0 = iy0 * width
    base1 = iy1 * width
    v00 = flat[base0 + ix0]
    v10 = flat[base0 + ix1]
    v01 = flat[base1 + ix0]
    v11 = flat[base1 + ix1]
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return float(top + (bottom - top) * ty)


class MeshWarp:
    """无状态的网格双线性变形器，可被多线程并发使用。"""

    def warp(self, image, grid, out_size):
        """按控制点网格对图像做双线性变形，返回 list[list[float]]。"""
        out_h, out_w = _validate_size(out_size)
        flat, height, width = _validate_image(image)
        _cells, nx, ny, xs, ys = _validate_grid(grid)

        # 输出像素中心在网格坐标系中的位置，坐标域取网格整体范围。
        x_min, x_max = xs[0], xs[-1]
        y_min, y_max = ys[0], ys[-1]
        span_x = x_max - x_min
        span_y = y_max - y_min
        out_xs = [
            x_min + (ox + 0.5) * span_x / out_w for ox in range(out_w)
        ]
        out_ys = [
            y_min + (oy + 0.5) * span_y / out_h for oy in range(out_h)
        ]

        width_m1 = width - 1
        height_m1 = height - 1

        def _axis_params(coords, knots, n_cells, pixel_span):
            """把每条输出线预解析为图像侧的 (下标0, 下标1, 插值权重)。

            网格定位（bisect）与 int 截断只做 O(行数+列数) 次，不进入像素
            热循环；控制点处权重为精确的 0/1，保证控制点精确。
            """
            params = []
            for f in coords:
                k = bisect_right(knots, f) - 1
                if k < 0:
                    k = 0
                elif k >= n_cells:
                    k = n_cells - 1
                t = (f - knots[k]) / (knots[k + 1] - knots[k])
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                pf = t * pixel_span
                p0 = int(pf)
                if p0 >= pixel_span:
                    p0 = p1 = pixel_span
                    w = 0.0
                else:
                    p1 = p0 + 1
                    w = pf - p0
                params.append((p0, p1, w))
            return params

        col_params = _axis_params(out_xs, xs, nx, width_m1)
        row_params = _axis_params(out_ys, ys, ny, height_m1)

        result = []
        for iy0, iy1, ty in row_params:
            base0 = iy0 * width
            base1 = iy1 * width
            row = []
            append = row.append
            for ix0, ix1, tx in col_params:
                v00 = flat[base0 + ix0]
                v10 = flat[base0 + ix1]
                v01 = flat[base1 + ix0]
                v11 = flat[base1 + ix1]
                top = v00 + (v10 - v00) * tx
                bottom = v01 + (v11 - v01) * tx
                append(top + (bottom - top) * ty)
            result.append(row)
        return result

    def sample(self, image, x, y):
        """对图像在归一化坐标 (x, y) 处双线性采样，越界 clamp。"""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("sample coordinates must be numbers")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("sample coordinates must be finite")
        flat, height, width = _validate_image(image)

        if x < 0.0:
            gx = 0.0
        elif x > 1.0:
            gx = 1.0
        else:
            gx = float(x)
        if y < 0.0:
            gy = 0.0
        elif y > 1.0:
            gy = 1.0
        else:
            gy = float(y)
        return _bilinear_sample(flat, width, height, gx, gy)
