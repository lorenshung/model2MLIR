"""State and ABI tests for Core ATen stochastic operators."""

from __future__ import annotations

import torch

import m2m
from m2m.coverage import CANONICAL_FAMILIES


class _Module(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _lower(fn, inputs=(), *, seed=17):
    result = m2m.convert(
        _Module(fn).eval(), tuple(inputs), backend="fx_importer", rng_seed=seed
    )
    assert result.ok
    assert "func.call" not in result.mlir_text
    return result


def test_rand_has_mutable_seed_and_one_state_advance():
    result = _lower(
        lambda: torch.ops.aten.rand.default([2, 3], dtype=torch.float32)
    )
    text = result.mlir_text
    assert "ml_program.global private mutable @global_seed(dense<17>" in text
    assert text.count("ml_program.global_load") == 1
    assert text.count("ml_program.global_store") == 1
    assert "6364136223846793005" in text
    assert "1442695040888963407" in text
    assert 'prov.rng_algorithm = "torch-mlir-lcg-hash-v1"' in text
    assert 'prov.aten = "aten.rand.default"' in text


def test_rng_seed_is_reproducible_and_configurable():
    fn = lambda: torch.ops.aten.rand.default([2, 3], dtype=torch.float32)
    first = _lower(fn, seed=23).mlir_text
    second = _lower(fn, seed=23).mlir_text
    other = _lower(fn, seed=24).mlir_text
    assert first == second
    assert first != other
    assert "@global_seed(dense<23>" in first
    assert "@global_seed(dense<24>" in other


def test_two_rand_calls_advance_the_same_state_twice():
    result = _lower(
        lambda: (
            torch.ops.aten.rand.default([2], dtype=torch.float32),
            torch.ops.aten.rand.default([2], dtype=torch.float32),
        )
    )
    text = result.mlir_text
    assert text.count("ml_program.global private mutable @global_seed") == 1
    assert text.count("ml_program.global_load") == 2
    assert text.count("ml_program.global_store") == 2


def test_randn_consumes_two_uniform_draws():
    result = _lower(
        lambda: torch.ops.aten.randn.default([2, 3], dtype=torch.float32)
    )
    text = result.mlir_text
    assert text.count("ml_program.global_load") == 2
    assert text.count("ml_program.global_store") == 2
    assert "math.log" in text
    assert "math.sqrt" in text
    assert "math.cos" in text
    assert 'prov.aten = "aten.randn.default"' in text


def test_randperm_returns_a_runtime_permutation_not_a_sample_constant():
    result = _lower(
        lambda: torch.ops.aten.randperm.default(6, dtype=torch.int64)
    )
    text = result.mlir_text
    assert "-> tensor<6xi64>" in text
    assert "dense<[0, 1, 2, 3, 4, 5]>" in text
    assert "arith.cmpf" in text
    assert "tensor.insert" in text
    assert text.count("ml_program.global_store") == 1
    assert 'prov.aten = "aten.randperm.default"' in text


def test_training_native_dropout_preserves_value_and_mask_abi():
    result = _lower(
        lambda x: torch.ops.aten.native_dropout.default(x, 0.25, True),
        (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
    )
    text = result.mlir_text
    assert "-> (tensor<2x3xf32>, tensor<2x3xi1>)" in text
    assert text.count("ml_program.global_store") == 1
    assert "arith.select" in text
    assert 'prov.aten = "aten.native_dropout.default"' in text


def test_nontraining_native_dropout_does_not_advance_state():
    result = _lower(
        lambda x: torch.ops.aten.native_dropout.default(x, 0.25, False),
        (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
    )
    text = result.mlir_text
    assert "-> (tensor<2x3xf32>, tensor<2x3xi1>)" in text
    assert "ml_program.global_store" not in text
    assert "dense<true>" not in text  # mask is created by a runtime tensor.splat


def test_random_is_an_explicit_canonical_family():
    assert "random" in CANONICAL_FAMILIES

