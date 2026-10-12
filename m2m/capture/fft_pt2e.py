"""Inspect constant-weight DFT candidates before PT2E, without target policy.

Candidates are deliberately separate from the source graph. A consumer must
admit every candidate operation before replacing an FFT; a matrix multiply
alone does not license component extraction, assembly, or layout changes.
"""

from __future__ import annotations

import math

import torch

from m2m.ir.decompositions import _twiddles

FFT_TARGETS = frozenset({"aten._fft_r2c.default", "aten._fft_c2r.default", "aten._fft_c2c.default"})


class ConstantDFT(torch.nn.Module):
    """The importer's separable DFT, expressed as observable ATen operations.

    Finite operands only: plain mm does not reproduce the importer's guarded
    zero-coefficient dependencies for nonfinite operands. Normalization and
    Hermitian multiplicity are folded into the existing twiddle constants.
    """

    def __init__(self, target, shape, dims, normalization, option):
        super().__init__()
        self.target = target
        self.shape = tuple(shape)
        self.dims = tuple(d % len(shape) for d in dims)
        if not self.dims or len(set(self.dims)) != len(self.dims):
            raise ValueError("FFT needs distinct transform dimensions")
        if normalization not in (0, 1, 2):
            raise ValueError("unknown FFT normalization")
        self.option = option
        lengths = [shape[d] for d in self.dims]
        if target == "aten._fft_c2r.default":
            lengths[-1] = int(option)
            if shape[self.dims[-1]] != int(option) // 2 + 1:
                raise ValueError("c2r input is not the requested half spectrum")
        if any(n <= 0 for n in lengths):
            raise ValueError("FFT transform lengths must be positive")
        total = math.prod(lengths)
        scale = (1.0, 1.0 / math.sqrt(total), 1.0 / total)[normalization]
        self.axes = self.dims if target == "aten._fft_c2r.default" else tuple(reversed(self.dims))
        current = list(shape)
        self.layouts = []
        for ordinal, axis in enumerate(self.axes):
            length = current[axis]
            n_out, period, fold = length, None, None
            if target == "aten._fft_r2c.default" and ordinal == 0 and option:
                n_out = length // 2 + 1
            if target == "aten._fft_c2r.default" and axis == self.dims[-1]:
                n_out = period = int(option)
                fold = [1.0 if b == 0 or (period % 2 == 0 and b == period // 2) else 2.0 for b in range(length)]
            sign = -1.0 if target == "aten._fft_r2c.default" or (target == "aten._fft_c2c.default" and option) else 1.0
            cosine, sine = _twiddles(
                length, n_out, sign=sign, scale=scale if ordinal == 0 else 1.0, period=period, fold=fold
            )
            self.register_buffer(f"cos_{ordinal}", torch.tensor(cosine, dtype=torch.float32))
            self.register_buffer(f"sin_{ordinal}", torch.tensor(sine, dtype=torch.float32))
            rest = [current[d] for d in range(len(current)) if d != axis]
            self.layouts.append((axis, length, n_out, tuple(rest)))
            current[axis] = n_out

    def forward(self, x):
        if self.target == "aten._fft_r2c.default":
            real, imag = x, None
        else:
            real, imag = x.real, x.imag
        for ordinal, (axis, length, n_out, rest) in enumerate(self.layouts):
            flat = len(self.shape) == 2 and axis == 1

            def flatten(value):
                return value if flat else value.movedim(axis, -1).reshape(math.prod(rest), length)

            def restore(value):
                return value if flat else value.reshape(*rest, n_out).movedim(-1, axis)

            a = flatten(real)
            cosine, sine = getattr(self, f"cos_{ordinal}"), getattr(self, f"sin_{ordinal}")
            rc = a @ cosine
            real_out = self.target == "aten._fft_c2r.default" and ordinal == len(self.layouts) - 1
            if imag is None:
                next_real, next_imag = rc, a @ sine
            else:
                b = flatten(imag)
                next_real = rc - b @ sine
                next_imag = None if real_out else b @ cosine + a @ sine
            real, imag = restore(next_real), None if next_imag is None else restore(next_imag)
        return real if self.target == "aten._fft_c2r.default" else torch.complex(real, imag)


def fft_candidates(exported_program):
    """Return exported per-site candidates and their complete ATen census.

    Uses metadata only, never reads source input values. No source graph,
    quantizer, calibration policy, or target declaration is changed.
    """
    candidates = []
    for node in exported_program.graph.nodes:
        target = str(node.target)
        if node.op != "call_function" or target not in FFT_TARGETS:
            continue
        source = node.args[0].meta["val"]
        if source.dtype not in (torch.float32, torch.complex64):
            raise ValueError(f"FFT pre-PT2E candidate requires f32/complex64, observed {source.dtype}")
        model = ConstantDFT(target, source.shape, *node.args[1:])
        # Fresh finite input is for shape/export only, never calibration.
        example = torch.zeros(tuple(source.shape), dtype=source.dtype)
        candidate = torch.export.export(model, (example,))
        rows = []
        for operation in candidate.graph.nodes:
            if operation.op != "call_function":
                continue
            operands = [
                arg.meta["val"] for arg in operation.all_input_nodes if isinstance(arg.meta.get("val"), torch.Tensor)
            ]
            result = operation.meta.get("val")
            rows.append(
                {
                    "node": operation.name,
                    "op": str(operation.target),
                    "operand_dtypes": [str(value.dtype) for value in operands],
                    "operand_ranks": [value.ndim for value in operands],
                    "constant_operands": [
                        arg.name in candidate.graph_signature.inputs_to_buffers
                        for arg in operation.all_input_nodes
                        if isinstance(arg.meta.get("val"), torch.Tensor)
                    ],
                    "result_dtype": str(result.dtype),
                    "rank": result.ndim,
                }
            )
        candidates.append(
            {
                "source_node": node.name,
                "source_op": target,
                "exported_program": candidate,
                "operations": rows,
                "finite_only": True,
            }
        )
    return candidates
