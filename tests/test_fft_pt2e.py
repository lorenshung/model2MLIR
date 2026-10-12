"""Candidate semantics and complete pre-PT2E operation visibility."""

import pytest
import torch
from m2m.capture.fft_pt2e import ConstantDFT, fft_candidates


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("normalization", [0, 1, 2])
@pytest.mark.parametrize("kind", ["r2c", "c2r", "c2c"])
def test_finite_candidate(n, normalization, kind):
    generator = torch.Generator().manual_seed(n)
    x = torch.randn(2, n, generator=generator)
    if kind == "c2r":
        x = torch.fft.rfft(x)
    elif kind == "c2c":
        x = torch.complex(x, torch.randn(2, n, generator=generator))
    option = n if kind == "c2r" else True
    target = f"aten._fft_{kind}.default"
    candidate = ConstantDFT(target, x.shape, [1], normalization, option)
    expected = getattr(torch.ops.aten, f"_fft_{kind}").default(x, [1], normalization, option)
    # DFT and FFT have different f32 accumulation orders. Bound cancellation
    # error by the input L1 norm, not a relative tolerance near zero bins.
    table_amax = max(buffer.abs().max().item() for buffer in candidate.buffers())
    tolerance = 8 * torch.finfo(torch.float32).eps * x.abs().sum(-1).max().item() * table_amax * n
    torch.testing.assert_close(candidate(x), expected, rtol=2e-5, atol=tolerance)


@pytest.mark.parametrize("kind", ["r2c", "c2r"])
def test_multiaxis_candidate(kind):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 24
    if kind == "c2r":
        x = torch.fft.rfftn(x, dim=(0, 2))
    option = 4 if kind == "c2r" else True
    target = f"aten._fft_{kind}.default"
    expected = getattr(torch.ops.aten, f"_fft_{kind}").default(x, [0, 2], 1, option)
    torch.testing.assert_close(ConstantDFT(target, x.shape, [0, 2], 1, option)(x), expected)


def test_candidate_does_not_mutate_source_and_exposes_assembly():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten._fft_r2c.default(x, [1], 0, True)

    source = torch.export.export(Model(), (torch.ones(2, 8),))
    before = str(source.graph)
    (candidate,) = fft_candidates(source)
    assert str(source.graph) == before
    assert [row["op"] for row in candidate["operations"]] == [
        "aten.matmul.default",
        "aten.matmul.default",
        "aten.complex.default",
    ]
    assert len(list(candidate["exported_program"].buffers())) == 2


def test_twosided_and_inverse_c2c():
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    torch.testing.assert_close(
        ConstantDFT("aten._fft_r2c.default", x.shape, [1], 2, False)(x),
        torch.ops.aten._fft_r2c.default(x, [1], 2, False),
    )
    z = torch.complex(x, x.flip(-1))
    torch.testing.assert_close(
        ConstantDFT("aten._fft_c2c.default", z.shape, [1], 2, False)(z),
        torch.ops.aten._fft_c2c.default(z, [1], 2, False),
    )
