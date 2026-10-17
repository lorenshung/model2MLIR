"""The pruned exact prefix-sum maximum equals the direct int64 scan it replaced, bit for bit."""

import pytest
import torch

from m2m.capture import pt2e_integer_reference as reference


def _original(lhs, rhs):
    """The tile loop exactly as it shipped before the pruned search."""
    if (
        lhs.ndim != 2
        or rhs.ndim != 2
        or lhs.shape[1] != rhs.shape[1]
        or min(*lhs.shape, *rhs.shape) < 1
    ):
        raise ValueError("partial-sum evidence has invalid contraction geometry")
    lhs, rhs = (
        lhs.to(device="cpu", dtype=torch.int64),
        rhs.to(device="cpu", dtype=torch.int64),
    )
    maximum = 0
    for row in range(0, lhs.shape[0], 64):
        left = lhs[row : row + 64]
        for col in range(0, rhs.shape[0], 32):
            right = rhs[col : col + 32]
            carry = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.int64)
            for start in range(0, left.shape[1], 128):
                products = (
                    left[:, None, start : start + 128]
                    * right[None, :, start : start + 128]
                )
                prefixes = products.cumsum(dim=-1) + carry[..., None]
                maximum = max(maximum, int(prefixes.abs().amax()))
                carry = prefixes[..., -1]
    return maximum


def _check(lhs, rhs):
    expected = _original(lhs, rhs)
    actual = reference._max_partial_sum(lhs, rhs)
    assert type(actual) is int
    assert actual == expected
    return actual


@pytest.mark.parametrize("seed", range(12))
def test_random_int8_and_centered_operands(seed):
    generator = torch.Generator().manual_seed(seed)
    shape = torch.randint(1, 80, (3,), generator=generator).tolist()
    rows, columns, depth = shape[0], shape[1], shape[2] * 7
    low = -255 if seed % 2 else -128
    high = 256 if seed % 2 else 128
    lhs = torch.randint(
        low, high, (rows, depth), generator=generator, dtype=torch.int64
    )
    rhs = torch.randint(
        -127, 128, (columns, depth), generator=generator, dtype=torch.int64
    )
    _check(lhs, rhs)


@pytest.mark.parametrize("seed", range(6))
def test_heavy_tailed_activations_with_outlier_channels(seed):
    generator = torch.Generator().manual_seed(100 + seed)
    rows, columns, depth = 70, 300, 1000 + 37 * seed
    lhs = (
        (torch.randn(rows, depth, generator=generator) * 6)
        .round()
        .clamp(-128, 127)
        .to(torch.int64)
    )
    outliers = torch.randint(0, depth, (5,), generator=generator)
    lhs[:, outliers] = torch.randint(-128, 128, (rows, 5), generator=generator)
    rhs = (
        (torch.randn(columns, depth, generator=generator) * 30)
        .round()
        .clamp(-127, 127)
        .to(torch.int64)
    )
    _check(lhs, rhs)


def test_the_maximum_inside_a_block_is_found():
    # Every block rises to 64 * 127 * 127 halfway and returns to zero: no block end holds the maximum.
    half = reference._PREFIX_BLOCK // 2
    row = torch.tensor(([127] * half + [-127] * half) * 5, dtype=torch.int64)
    lhs = torch.stack([row, -row, torch.zeros_like(row), row // 2, -row // 3])
    assert lhs.shape[0] >= reference._DIRECT_ROWS
    rhs = torch.full((4, row.numel()), 127, dtype=torch.int64)
    assert _check(lhs, rhs) == half * 127 * 127


@pytest.mark.parametrize("value", [-128, 127, 255, -255])
def test_constant_operands_past_the_i32_range(value):
    depth = (1 << 31) // (127 * 127) + 3 * reference._PREFIX_BLOCK + 5
    lhs = torch.full((5, depth), value, dtype=torch.int64)
    lhs[3] = -value
    rhs = torch.full((3, depth), -128, dtype=torch.int64)
    rhs[1] = 127
    result = _check(lhs, rhs)
    assert result > (1 << 31) - 1


def test_operands_on_both_sides_of_the_float64_bound():
    depth = 4
    exact = reference._EXACT_FLOAT64_BOUND
    below = int((exact // depth) ** 0.5) - 1
    for magnitude in (below, below + 2, 1 << 30, (1 << 31) - 1):
        lhs = torch.tensor(
            [[magnitude, -magnitude, magnitude, 1], [1, 2, 3, 4]] * 3, dtype=torch.int64
        )
        rhs = torch.tensor(
            [[magnitude, magnitude, magnitude, -magnitude]], dtype=torch.int64
        )
        _check(lhs, rhs)


def test_wrapping_int64_operands_match_the_direct_scan():
    # Products past int64 wrap in both implementations; the bound sends these to the direct scan.
    big = torch.tensor([[1 << 40, -(1 << 40), 1 << 40]] * 5, dtype=torch.int64)
    _check(big, big)
    edge = torch.tensor([[-(1 << 63), 1, (1 << 63) - 1]], dtype=torch.int64)
    _check(edge, torch.ones((2, 3), dtype=torch.int64))


def test_float_valued_and_int32_operands_convert_like_the_original():
    generator = torch.Generator().manual_seed(7)
    lhs = torch.randint(-128, 128, (9, 200), generator=generator).to(torch.int32)
    rhs = torch.randint(-127, 128, (11, 200), generator=generator).to(torch.float32)
    _check(lhs, rhs)


def test_large_tiles_and_few_rows():
    generator = torch.Generator().manual_seed(11)
    lhs = torch.randint(-128, 128, (1, 2048), generator=generator, dtype=torch.int64)
    rhs = torch.randint(-127, 128, (3000, 2048), generator=generator, dtype=torch.int64)
    _check(lhs, rhs)
    for rows in range(1, reference._DIRECT_ROWS + 3):
        lhs = torch.randint(
            -128, 128, (rows, 700), generator=generator, dtype=torch.int64
        )
        _check(lhs, rhs[:, :700])
    lhs = torch.randint(-128, 128, (130, 2048), generator=generator, dtype=torch.int64)
    rhs = torch.randint(-127, 128, (70, 2048), generator=generator, dtype=torch.int64)
    _check(lhs, rhs)


def test_invalid_geometry_is_refused():
    with pytest.raises(ValueError, match="geometry"):
        reference._max_partial_sum(torch.ones((2, 3)), torch.ones((2, 4)))
    with pytest.raises(ValueError, match="geometry"):
        reference._max_partial_sum(torch.ones((0, 3)), torch.ones((2, 3)))
