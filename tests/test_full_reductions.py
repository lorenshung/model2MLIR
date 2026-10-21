"""Full reductions must visit every element through native MLIR execution."""

import pytest
import torch

from native_execution import assert_native_matches_eager


REDUCTIONS = {
    "sum_default": lambda x: torch.ops.aten.sum.default(x),
    "sum_none": lambda x: torch.ops.aten.sum.dim_IntList(x, None),
    "sum_empty": lambda x: torch.ops.aten.sum.dim_IntList(x, []),
    "sum_empty_keep": lambda x: torch.ops.aten.sum.dim_IntList(x, [], True),
    "sum_partial": lambda x: x.sum(dim=-1, keepdim=True),
    "mean_default": lambda x: x.mean(),
    "mean_empty": lambda x: x.mean(dim=[]),
    "mean_empty_keep": lambda x: x.mean(dim=[], keepdim=True),
    "mean_partial": lambda x: x.mean(dim=-1, keepdim=True),
    "prod_default": lambda x: x.prod(),
    "prod_partial": lambda x: x.prod(dim=-1, keepdim=True),
    "amax_empty": lambda x: x.amax(dim=[]),
    "amin_empty_keep": lambda x: x.amin(dim=[], keepdim=True),
    "amax_partial": lambda x: x.amax(dim=-1, keepdim=True),
    "min_default": lambda x: x.min(),
    "max_default": lambda x: x.max(),
    "var_empty": lambda x: x.var(dim=[], correction=0),
    "var_none_keep": lambda x: x.var(dim=None, correction=0, keepdim=True),
    "var_partial": lambda x: x.var(dim=-1, correction=0, keepdim=True),
    "var_mean_empty": lambda x: torch.var_mean(x, dim=[], correction=0, keepdim=True),
    "std_default": lambda x: x.std(correction=0),
    "any_default": lambda x: x.any(),
    "all_default": lambda x: x.all(),
    # Unlike sum/mean/amin/amax, boolean reductions leave an empty dim list
    # unreduced (but still convert values to truth values).
    "any_empty": lambda x: torch.any(x, dim=[]),
    "all_empty": lambda x: torch.all(x, dim=[]),
    "any_partial": lambda x: x.any(dim=-1, keepdim=True),
}


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "padded", "expanded"])
@pytest.mark.parametrize("name", REDUCTIONS)
def test_native_full_and_partial_reductions(tmp_path, layout, name):
    x = torch.tensor([[0., 1., 2.], [3., 4., 5.]]) / 4
    if layout == "transposed":
        x = x.T.contiguous().T
    elif layout == "padded":
        storage = torch.empty(32)
        x = storage.as_strided((2, 3), (7, 2), 3).copy_(x)
    elif layout == "expanded":
        x = x[:1].expand(2, 3)
    if name.startswith(("any_", "all_")):
        x = x.bool()
    assert_native_matches_eager(tmp_path, REDUCTIONS[name], (x,))


@pytest.mark.parametrize("name", [
    "sum_default", "sum_empty_keep", "sum_partial", "prod_default", "prod_partial",
])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64])
def test_native_integer_accumulator_promotion(tmp_path, name, dtype):
    value = 250 if dtype == torch.uint8 else 120
    x = torch.tensor([[value, value, value], [1, 2, 3]], dtype=dtype)
    assert_native_matches_eager(tmp_path, REDUCTIONS[name], (x,))


@pytest.mark.parametrize("name", [
    "sum_default", "sum_empty_keep", "sum_partial", "mean_default", "mean_empty_keep",
    "mean_partial", "prod_default", "prod_partial", "var_empty", "std_default",
    "any_default", "all_default", "any_empty", "all_empty",
])
def test_native_empty_identities(tmp_path, name):
    dtype = torch.bool if name.startswith(("any_", "all_")) else torch.float32
    assert_native_matches_eager(tmp_path, REDUCTIONS[name], (torch.empty(2, 0, dtype=dtype),))


@pytest.mark.parametrize("name", REDUCTIONS)
def test_native_scalar_reductions(tmp_path, name):
    x = torch.tensor(True) if name.startswith(("any_", "all_")) else torch.tensor(2.)
    assert_native_matches_eager(tmp_path, REDUCTIONS[name], (x,))


@pytest.mark.parametrize("fn", [
    lambda x: x.sum(dtype=torch.int64),
    lambda x: x.sum(dim=[], keepdim=True, dtype=torch.float64),
    lambda x: x.mean(dim=[], keepdim=True, dtype=torch.float64),
    lambda x: x.mean(dtype=torch.float64),
    lambda x: x.prod(dtype=torch.int64),
])
def test_native_explicit_accumulator_dtype(tmp_path, fn):
    assert_native_matches_eager(tmp_path, fn, (torch.full((2, 3), 120, dtype=torch.int8),))


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int32])
@pytest.mark.parametrize("name", ["amax_empty", "amin_empty_keep", "min_default", "max_default"])
def test_native_extremum_integer_order(tmp_path, dtype, name):
    x = torch.tensor([[0, 250, 2], [3, 4, 5]], dtype=torch.uint8).to(dtype)
    assert_native_matches_eager(tmp_path, REDUCTIONS[name], (x,))


@pytest.mark.parametrize("name,fn,x", [
    *((name, REDUCTIONS[name], torch.ones(4096, dtype=torch.float16))
      for name in ["sum_default", "mean_default", "var_empty", "std_default", "var_mean_empty"]),
    ("prod", lambda x: x.prod(), torch.full((64,), 1.001, dtype=torch.float16)),
    # An explicit half dtype casts before accumulating.
    ("sum_half_cast", lambda x: x.sum(dtype=torch.float16), torch.full((4096,), 1.0001, dtype=torch.float32))])
def test_native_half_accumulator(tmp_path, name, fn, x):
    assert_native_matches_eager(tmp_path, fn, (x,))


@pytest.mark.parametrize("fn", [
    lambda x: x.logsumexp([0, 1]),
    lambda x: x.logsumexp([-1], keepdim=True),
    lambda x: x.sum(dim=[-1, 0], keepdim=True),
    lambda x: x.mean(dim=[-1, 0], keepdim=True),
    lambda x: x.argmax(dim=-1, keepdim=True),
    lambda x: x.argmin(dim=-1, keepdim=True),
    lambda x: x.max(dim=-1, keepdim=True),
    lambda x: x.min(dim=-1, keepdim=True),
])
def test_native_related_reductions(tmp_path, fn):
    x = torch.tensor([[0., 1., 1.], [-2., -2., 3.]]).T.contiguous().T
    assert_native_matches_eager(tmp_path, fn, (x,))


@pytest.mark.parametrize("fn", [
    lambda x: x.argmax(dim=-1, keepdim=True),
    lambda x: x.argmin(dim=0),
    lambda x: x.max(dim=0, keepdim=True),
    lambda x: x.min(dim=-1),
])
def test_native_scalar_dim_extrema(tmp_path, fn):
    assert_native_matches_eager(tmp_path, fn, (torch.tensor(2.),))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.float32, None])
@pytest.mark.parametrize("fn", [
    lambda x: x.sum(dtype=torch.bool),
    lambda x: x.sum(dim=[], keepdim=True, dtype=torch.bool),
    lambda x: x.sum(dim=-1, keepdim=True, dtype=torch.bool),
])
def test_native_explicit_boolean_sum(tmp_path, fn, dtype):
    x = (torch.empty(2, 0) if dtype is None else
         torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=dtype))
    assert_native_matches_eager(tmp_path, fn, (x,))
