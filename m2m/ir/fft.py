"""Real FFT spectrum completion over the existing real/imaginary component ABI."""

from xdsl.dialects import arith, tensor

from m2m.ir.structural import generate, index


def complete_hermitian(half, shape, dims, elem):
    """Copy the stored half and conjugate the missing multidimensional bins.

    A negative last-axis frequency reflects every transformed axis. The
    nontransformed batch coordinates stay unchanged. Negation, rather than a
    second DFT or multiplication, preserves conjugate signed zeros and NaNs.
    """
    last = dims[-1]

    def build(block, coords):
        limit = index(shape[last] // 2)
        mirrored = arith.CmpiOp(coords[last], limit.result, "ugt")
        block.add_ops([limit, mirrored])
        source = list(coords)
        for axis in dims:
            size = index(shape[axis])
            backwards = arith.SubiOp(size.result, coords[axis])
            wrapped = arith.RemUIOp(backwards.result, size.result)
            selected = arith.SelectOp(mirrored.result, wrapped.result, coords[axis])
            block.add_ops([size, backwards, wrapped, selected])
            source[axis] = selected.result
        value = tensor.ExtractOp(half, source, elem)
        one = index(1)
        imaginary = arith.CmpiOp(coords[-1], one.result, "eq")
        conjugate = arith.AndIOp(mirrored.result, imaginary.result)
        negative = arith.NegfOp(value.result)
        result = arith.SelectOp(conjugate.result, negative.result, value.result)
        block.add_ops([value, one, imaginary, conjugate, negative, result])
        return result.result

    return generate([*shape, 2], elem, build)
