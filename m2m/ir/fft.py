"""Real FFT spectrum completion over the existing real/imaginary component ABI."""

from xdsl.dialects import arith, tensor

from m2m.ir.builder import Builder
from m2m.ir.structural import generate


def complete_hermitian(half, shape, dims, elem):
    """Copy the stored half and conjugate the missing multidimensional bins.

    A negative last-axis frequency reflects every transformed axis. The
    nontransformed batch coordinates stay unchanged. Negation, rather than a
    second DFT or multiplication, preserves conjugate signed zeros and NaNs.
    """
    last = dims[-1]

    def build(block, coords):
        b = Builder(block)
        mirrored = b(arith.CmpiOp(coords[last], b.ix(shape[last] // 2), "ugt"))
        source = list(coords)
        for axis in dims:
            size = b.ix(shape[axis])
            wrapped = b(arith.RemUIOp(b(arith.SubiOp(size, coords[axis])), size))
            source[axis] = b(arith.SelectOp(mirrored, wrapped, coords[axis]))
        value = b(tensor.ExtractOp(half, source, elem))
        conjugate = b(arith.AndIOp(mirrored, b(arith.CmpiOp(coords[-1], b.ix(1), "eq"))))
        return b(arith.SelectOp(conjugate, b(arith.NegfOp(value)), value))

    return generate([*shape, 2], elem, build)
