from types import SimpleNamespace

import pytest
import sympy
import torch
from torch.utils._sympy.numbers import int_oo

from m2m.capture.torch_export import (
    RangeConstraint,
    _prepare_exported_program,
    _serialize_range_constraints,
)


def _export_with_constraints(constraints):
    return SimpleNamespace(range_constraints=constraints)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-sympy.oo, sympy.oo),
        (-int_oo, int_oo),
        (float("-inf"), float("inf")),
    ],
)
def test_serialize_range_constraints_maps_infinite_bounds_to_none(lower, upper):
    exported = _export_with_constraints(
        {sympy.Symbol("unbounded"): SimpleNamespace(lower=lower, upper=upper)}
    )

    assert _serialize_range_constraints(exported) == (
        RangeConstraint(symbol="unbounded", minimum=None, maximum=None),
    )


def test_serialize_range_constraints_preserves_finite_bounds():
    exported = _export_with_constraints(
        {
            sympy.Symbol("bounded"): SimpleNamespace(
                lower=sympy.Integer(-7), upper=sympy.Integer(11)
            )
        }
    )

    assert _serialize_range_constraints(exported) == (
        RangeConstraint(symbol="bounded", minimum=-7, maximum=11),
    )


def test_cosh_decomposition_serializes_unbounded_shape_constraints():
    class Cosh(torch.nn.Module):
        def forward(self, value):
            return torch.ops.aten.cosh.default(value)

    # A view input makes export retain storage-shape symbols.  Decomposing
    # cosh gives those symbols an ``int_oo`` upper bound in the pinned PyTorch.
    view_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original = torch.export.export(Cosh(), (view_input,))
    prepared, _ = _prepare_exported_program(
        original,
        run_default_decompositions=False,
        export_decomposition_table={
            torch.ops.aten.cosh.default: torch._decomp.decomposition_table[
                torch.ops.aten.cosh.default
            ]
        },
    )

    constraints = _serialize_range_constraints(prepared)

    assert constraints
    assert all(constraint.maximum is None for constraint in constraints)
    assert all(constraint.minimum == 2 for constraint in constraints)
