"""CPU grid sampling with native coordinate rounding and absent edge handling."""

import torch


def grid_sampler_2d_cpu(input, grid, interpolation_mode=0, padding_mode=0, align_corners=False):
    from torch._decomp import decomposition_table

    if input.device.type != "cpu" or interpolation_mode not in (0, 1, 2):
        return decomposition_table[torch.ops.aten.grid_sampler_2d.default](
            input, grid, interpolation_mode, padding_mode, align_corners
        )
    n, c, height, width = input.shape
    oh, ow = grid.shape[1:3]

    def bounds(value, size):
        if padding_mode == 1:
            # Native border clipping maps NaNs to the low boundary.
            value = torch.where(torch.isnan(value), 0.0, value).clamp(0, size - 1)
        elif padding_mode == 2:
            low, span = (0.0, float(size - 1)) if align_corners else (-0.5, float(size))
            if span == 0:
                return torch.zeros_like(value)
            distance = (value - low).abs()
            period = 2 * span
            turns = (distance / period).trunc()
            extra = distance - turns * period
            value = torch.minimum(extra, period - extra) + low
            value = torch.where(torch.isnan(value), 0.0, value).clamp(0, size - 1)
        return value

    def unnormalize(value, size):
        # Native scalar_t observes rounding after each operation.
        return (value + 1) * ((size - 1) / 2) if align_corners else (value + 1) * (size / 2) - 0.5

    x = unnormalize(grid[..., 0], width).unsqueeze(1)
    y = unnormalize(grid[..., 1], height).unsqueeze(1)
    if interpolation_mode != 2:
        x, y = bounds(x, width), bounds(y, height)
    batch = torch.arange(n).reshape(n, 1, 1, 1)
    channel = torch.arange(c).reshape(1, c, 1, 1)

    def sample(ix, iy, factor):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        safe_x = torch.where(valid, ix, 0).to(torch.int64)
        safe_y = torch.where(valid, iy, 0).to(torch.int64)
        value = input[batch, channel, safe_y, safe_x]
        # A missing neighbor is an absent contribution. Using pixel (0,0)
        # with a zero coefficient still contaminates the result if it is NaN.
        return torch.where(valid, value * factor, 0.0)

    if interpolation_mode == 2:
        xbase, ybase = x.floor(), y.floor()
        tx, ty = x - xbase, y - ybase

        def cubic(values, fraction):
            alpha = -0.75

            def near(value):
                return ((alpha + 2) * value - (alpha + 3)) * value * value + 1

            def far(value):
                return ((alpha * value - 5 * alpha) * value + 8 * alpha) * value - 4 * alpha

            weights = [far(fraction + 1), near(fraction), near(1 - fraction), far(2 - fraction)]
            return values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2] + values[3] * weights[3]

        rows = []
        for row in range(4):
            values = [
                sample(bounds(position, width), bounds(ybase - 1 + row, height), 1.0)
                for position in (xbase - 1, xbase, xbase + 1, xbase + 2)
            ]
            rows.append(cubic(values, tx))
        return cubic(rows, ty)
    if interpolation_mode == 1:
        return sample(x.round(), y.round(), 1.0)
    x0, y0 = x.floor(), y.floor()
    x1, y1 = x0 + 1, y0 + 1
    result = (
        sample(x0, y0, (x1 - x) * (y1 - y))
        + sample(x1, y0, (x - x0) * (y1 - y))
        + sample(x0, y1, (x1 - x) * (y - y0))
        + sample(x1, y1, (x - x0) * (y - y0))
    )

    if padding_mode == 0:
        invalid = torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y)
        result = torch.where(invalid, float("nan"), result)
    return result
