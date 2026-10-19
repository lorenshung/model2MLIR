"""Portable expression of the source CPU's floating NaN and SIMD policies.

NaN word selection is explicit, so a target's canonical NaN cannot change the
captured CPU storage result. Finite arithmetic remains ordinary IEEE MLIR.
"""
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType


def cast(put, value, target):
    """Source CPU float conversion with explicit NaN bits across target CPUs."""
    from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, Float64Type
    from .decompositions import _cast_scalar_arg
    source = value.type
    if (isinstance(target, (BFloat16Type, Float16Type)) and source != target
            and isinstance(source, (Float64Type, BFloat16Type, Float16Type, IntegerType))):
        # CPU reduced storage conversion passes through float, also for a
        # weak zero-dimensional Tensor of another floating dtype.
        return cast(put, cast(put, value, Float32Type()), target)
    ops, converted = _cast_scalar_arg(value, target)
    for op in ops:
        put(op)
    fractions = {Float16Type(): 10, BFloat16Type(): 7, Float32Type(): 23, Float64Type(): 52}
    if source == target or source not in fractions or target not in fractions:
        return converted
    integer = IntegerType(target.bitwidth)
    original = put(arith.BitcastOp(value, IntegerType(source.bitwidth)))
    if target.bitwidth > source.bitwidth:
        original = put(arith.ExtUIOp(original, integer))
    sign = put(arith.ShRUIOp(original, put(arith.ConstantOp(IntegerAttr(source.bitwidth - 1, original.type)))))
    if sign.type != integer:
        sign = put(arith.TruncIOp(sign, integer))
    sign = put(arith.ShLIOp(sign, put(arith.ConstantOp(IntegerAttr(target.bitwidth - 1, integer)))))
    mask = put(arith.ConstantOp(IntegerAttr((1 << fractions[source]) - 1, original.type)))
    payload = put(arith.AndIOp(original, mask))
    shift = fractions[target] - fractions[source]
    payload = put((arith.ShLIOp if shift > 0 else arith.ShRUIOp)(
        payload, put(arith.ConstantOp(IntegerAttr(abs(shift), payload.type))))) if shift else payload
    if payload.type != integer:
        payload = put(arith.TruncIOp(payload, integer))
    exponent = ((1 << (target.bitwidth - fractions[target] - 1)) - 1) << fractions[target]
    quiet = exponent | (1 << (fractions[target] - 1))
    bits = put(arith.OrIOp(sign, payload))
    bits = put(arith.OrIOp(bits, put(arith.ConstantOp(IntegerAttr(quiet, integer)))))
    if isinstance(target, BFloat16Type):
        bits = put(arith.ConstantOp(IntegerAttr(0x7fc0, integer)))
    nan = put(arith.BitcastOp(bits, target))
    missing = put(arith.CmpfOp(value, value, 'uno'))
    return put(arith.SelectOp(missing, nan, converted))


def nan_result(put, result, operands):
    """x86 arithmetic: first input NaN (quieted), else negative invalid NaN."""
    width = result.type.bitwidth
    integer = IntegerType(width)
    from xdsl.dialects.builtin import BFloat16Type
    fraction = (7 if isinstance(result.type, BFloat16Type) else 10) if width == 16 else (23 if width == 32 else 52)
    quiet = 1 << (fraction - 1)
    invalid = (1 << width) - (1 << fraction) + quiet
    constant = put(arith.ConstantOp(IntegerAttr(invalid - (1 << width), integer)))
    nan = put(arith.BitcastOp(constant, result.type))
    unordered = put(arith.CmpfOp(result, result, 'uno'))
    chosen = put(arith.SelectOp(unordered, nan, result))
    mask = put(arith.ConstantOp(IntegerAttr(quiet, integer)))
    for value in reversed(operands):
        bits = put(arith.BitcastOp(value, integer))
        bits = put(arith.OrIOp(bits, mask))
        payload = put(arith.BitcastOp(bits, result.type))
        missing = put(arith.CmpfOp(value, value, 'uno'))
        chosen = put(arith.SelectOp(missing, payload, chosen))
    return chosen


def binary(put, cls, a, b):
    return nan_result(put, put(cls(a, b)), [a, b])


def negate_alpha(value):
    """Scalar::operator- negates its integral word and returns a signed Scalar."""
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 64):
            raise ValueError('CPU add/sub requires a representable 64-bit Scalar')
        return ((-value + (1 << 63)) % (1 << 64)) - (1 << 63)
    return -value


def check_coefficient(value, target):
    """Scalar::to rejects finite alpha overflow before rounding to storage."""
    import math
    from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, Float64Type
    precision, exponent = {Float16Type(): (11, 15), BFloat16Type(): (8, 127),
                           Float32Type(): (24, 127), Float64Type(): (53, 1023)}[target]
    maximum = ((1 << precision) - 1) << (exponent - precision + 1)
    if not (isinstance(value, float) and math.isinf(value)) and abs(value) > maximum:
        raise ValueError('CPU add/sub alpha cannot convert to its scalar dtype without overflow')


def scalar(put, value, target, *, checked=False):
    """Scalar::to preserves integer coefficients and the reduced f32 bridge."""
    from xdsl.dialects.builtin import BFloat16Type, Float16Type, Float32Type, FloatAttr, i64
    from .quant_attrs import fp32
    if checked:
        check_coefficient(value, target)
    computation = Float32Type() if isinstance(target, (BFloat16Type, Float16Type)) else target
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 64):
            raise ValueError('CPU add/sub requires a representable 64-bit Scalar')
        unsigned = value >= (1 << 63)
        integer = put(arith.ConstantOp(IntegerAttr(value - (1 << 64) if unsigned else value, i64)))
        converted = put((arith.UIToFPOp if unsigned else arith.SIToFPOp)(integer, computation))
    else:
        literal = fp32(value) if isinstance(computation, Float32Type) else float(value)
        converted = put(arith.ConstantOp(FloatAttr(literal, computation)))
    return cast(put, converted, target)


def _source(meta, operand):
    """Metadata of TensorIterator's common-dtype input temporary."""
    import torch
    from .decompositions import _fx_arg
    arg = _fx_arg(meta, operand)
    source = getattr(arg, 'meta', {}).get('val', arg)
    if isinstance(source, torch.Tensor) and source.dtype != meta['val'].dtype:
        # TensorIterator.cpp compute_types promotes with tensor.to(dtype).
        # empty_like on meta has the same preserve-format allocation policy,
        # including densifying expanded/padded inputs, without reading data.
        source = torch.empty_like(source, dtype=meta['val'].dtype, device='meta')
    return source


def scalar_run(meta, operand):
    """Whether an input broadcasts within the TensorIterator inner row."""
    source = _source(meta, operand)
    if _unit_fast_setup(meta):
        return False
    if not hasattr(source, 'shape'):
        return True
    output = meta['val']
    source = source.expand(output.shape)
    axes = sorted(range(output.ndim), key=lambda a: output.stride(a))
    axis = next((a for a in axes if output.shape[a] > 1), axes[0] if axes else None)
    return axis is None or source.stride(axis) == 0


def rhs_scalar_run(meta):
    return scalar_run(meta, 1)


def _unit_fast_setup(meta):
    """All-same-shape unit tensors use TensorIterator's contiguous setup."""
    output = meta['val']
    if output.numel() != 1:
        return False
    for i in (0, 1):
        source = _source(meta, i)
        shape = source.shape if hasattr(source, 'shape') else ()
        if tuple(shape) != tuple(output.shape):
            return False
    return True


def _inner_run(meta):
    import torch
    from .cpu_pointwise import inner_run
    output = meta['val']
    if torch.backends.cpu.get_cpu_capability() != 'AVX2' or output.device.type != 'cpu':
        raise ValueError('CPU add/sub storage policy requires the verified AVX2 source kernels')
    size, axes, vectorizable = output.numel(), None, True
    sources = []
    for i in (0, 1):
        source = _source(meta, i)
        if hasattr(source, 'shape'):
            source = source.expand(output.shape)
            sources.append(source)
    for source in sources:
        if _unit_fast_setup(meta):
            count, valid, order = 1, True, []
        elif size == 1 and output.ndim:
            # TensorIterator keeps the outermost unit dimension's strides
            # when coalescing an all-singleton shape.
            order = [0]
            count = 1
            valid = source.stride(0) in (0, 1) and output.stride(0) == 1
        else:
            count, valid, order = inner_run(source, output)
        if count <= size:
            size, axes = count, order
        vectorizable &= valid
    if axes is None:
        axes = []
    # Loops.h permits at most one scalar input in vectorized_loop.
    vectorizable &= not all(scalar_run(meta, i) for i in (0, 1))
    if '_cpu_segment' in meta:
        size = meta['_cpu_segment'][1] - meta['_cpu_segment'][0]
    return size, axes, vectorizable


def _position(put, meta, axes):
    output = meta['val']
    segment = meta.get('_cpu_segment')
    if segment is not None:
        axes = sorted(range(output.ndim), key=lambda a: output.stride(a))
    position = put(arith.ConstantOp(IntegerAttr(0, IndexType())))
    factor = 1
    for axis in axes:
        coord = put(linalg.IndexOp(axis))
        stride = put(arith.ConstantOp(IntegerAttr(factor, IndexType())))
        position = put(arith.AddiOp(position, put(arith.MuliOp(coord, stride))))
        factor *= output.shape[axis]
    if segment is not None:
        start = put(arith.ConstantOp(IntegerAttr(segment[0], IndexType())))
        position = put(arith.SubiOp(position, start))
    return position


def _thread_slices(profile):
    """Restart the CPU loop policy at TensorIterator parallel range boundaries.

    ParallelOpenMP.h splits the flattened iterator into ceil(N / tasks) ranges,
    with tasks=min(threads, ceil(N / GRAIN_SIZE)). DimCounter in
    TensorIteratorInternal.h intersects each range with the coalesced inner
    rows. Only the first/last partial rows differ from a complete row; retain
    the ordinary policy everywhere else. This describes the source traversal,
    independently of the target's execution threads.
    """
    from functools import wraps

    @wraps(profile)
    def sliced(put, meta, *args, **kwargs):
        import torch
        if meta.get('_cpu_profile'):
            return profile(put, meta, *args, **kwargs)
        result = profile(put, {**meta, '_cpu_profile': True}, *args, **kwargs)
        total = meta['val'].numel()
        threads = torch.get_num_threads()
        if total <= 32768 or threads == 1:
            return result
        tasks = min(threads, (total + 32767) // 32768)
        chunk = (total + tasks - 1) // tasks
        row, axes, _ = _inner_run(meta)
        if not row:
            return result
        segments = set()
        for start in range(0, total, chunk):
            end = min(total, start + chunk)
            first_end = min(end, (start // row + 1) * row)
            if first_end - start < row:
                segments.add((start, first_end))
            last_start = max(start, (end - 1) // row * row)
            if end - last_start < row:
                segments.add((last_start, end))
        if not segments:
            return result
        axes = sorted(range(meta['val'].ndim), key=lambda a: meta['val'].stride(a))
        position = _position(put, meta, axes)
        for start, end in sorted(segments):
            local = profile(put, {**meta, '_cpu_profile': True, '_cpu_segment': (start, end)}, *args, **kwargs)
            lower = put(arith.ConstantOp(IntegerAttr(start, IndexType())))
            upper = put(arith.ConstantOp(IntegerAttr(end, IndexType())))
            selected = put(arith.AndIOp(put(arith.CmpiOp(position, lower, 'uge')),
                                       put(arith.CmpiOp(position, upper, 'ult'))))
            if isinstance(result, tuple):
                result = tuple(put(arith.SelectOp(selected, a, b)) for a, b in zip(local, result, strict=True))
            else:
                result = put(arith.SelectOp(selected, local, result))
        return result

    return sliced


@_thread_slices
def simd_predicate(put, meta, block_size, *, strided=False, only_strided=False):
    """Full vector prefix of each coalesced TensorIterator row, from metadata."""
    size, axes, vectorizable = _inner_run(meta)
    enabled = (not vectorizable) if only_strided else (vectorizable or strided)
    prefix = size // block_size * block_size if enabled else 0
    position = _position(put, meta, axes)
    limit = put(arith.ConstantOp(IntegerAttr(prefix, IndexType())))
    return put(arith.CmpiOp(position, limit, 'ult'))


@_thread_slices
def scalar_tail_blocks(put, meta, width, operand):
    """GCC's scalar-broadcast tail: peel one FMA, then SSE blocks.

    After the explicit two-AVX-vector prefix, basic_loop processes four float
    / two double elements at a time. A broadcast RHS hoists alpha*other;
    a broadcast LHS keeps FMA with a different NaN operand order.
    """
    size, axes, vectorizable = _inner_run(meta)
    prefix = size // (512 // width) * (512 // width)
    block = 128 // width
    start = prefix + 1
    end = start + max(0, size - start) // block * block
    if not vectorizable or not scalar_run(meta, operand):
        end = start
    position = _position(put, meta, axes)
    lower = put(arith.ConstantOp(IntegerAttr(start, IndexType())))
    upper = put(arith.ConstantOp(IntegerAttr(end, IndexType())))
    return put(arith.AndIOp(put(arith.CmpiOp(position, lower, 'uge')),
                           put(arith.CmpiOp(position, upper, 'ult'))))


@_thread_slices
def reduced_other_nan_first(put, meta):
    """AVX2 Half RHS broadcast swaps the final float FMA's registers.

    The installed add kernel uses three vfmadd231ps instructions, then a
    vfmadd132ps with alpha in the destination for the final eight lanes.
    That reverses the two multiplicands' NaN priority without changing FMA.
    """
    from xdsl.dialects.builtin import i1
    if not rhs_scalar_run(meta):
        return put(arith.ConstantOp(IntegerAttr(1, i1)))
    _, axes, _ = _inner_run(meta)
    position = _position(put, meta, axes)
    block = put(arith.ConstantOp(IntegerAttr(32, IndexType())))
    lane = put(arith.RemUIOp(position, block))
    end = put(arith.ConstantOp(IntegerAttr(24, IndexType())))
    return put(arith.CmpiOp(lane, end, 'ult'))


@_thread_slices
def real_other_nan_first(put, meta, width):
    """Operand order of AVX2 add's FMA instructions, including scalar unrolls."""
    from xdsl.dialects.builtin import i1
    size, axes, vectorizable = _inner_run(meta)
    if not vectorizable:
        # The generic GCC loop uses a different register order from its
        # versioned double loop (enabled at 15 elements), including its tail.
        return put(arith.ConstantOp(IntegerAttr(int(width == 32 or size >= 15), i1)))
    if rhs_scalar_run(meta):
        return put(arith.ConstantOp(IntegerAttr(1, i1)))
    simd = simd_predicate(put, meta, 512 // width)
    if scalar_run(meta, 0):
        prefix = size // (512 // width) * (512 // width)
        remainder = size - prefix
        if width == 32:
            start = prefix + 1 + max(0, remainder - 1) // 4 * 4
            count = max(0, size - start)
        else:
            start = size - 1
            count = int(remainder >= 4 and remainder % 2 == 0)
        position = _position(put, meta, axes)
        lower = put(arith.ConstantOp(IntegerAttr(start, IndexType())))
        upper = put(arith.ConstantOp(IntegerAttr(start + count, IndexType())))
        tail = put(arith.AndIOp(put(arith.CmpiOp(position, lower, 'uge')),
                              put(arith.CmpiOp(position, upper, 'ult'))))
        return put(arith.OrIOp(simd, tail))
    position = _position(put, meta, axes)
    if width == 32:
        # SSE blocks of four keep alpha first. The final scalar unroll
        # keeps other first in its first two slots, alpha in its third.
        start = size // 4 * 4
        count = min(2, size % 4)
    else:
        # A lone double tail keeps other first; longer tails keep alpha.
        start = size // 8 * 8
        count = 1 if size % 8 == 1 else 0
    lower = put(arith.ConstantOp(IntegerAttr(start, IndexType())))
    upper = put(arith.ConstantOp(IntegerAttr(start + count, IndexType())))
    tail = put(arith.AndIOp(put(arith.CmpiOp(position, lower, 'uge')),
                          put(arith.CmpiOp(position, upper, 'ult'))))
    return put(arith.OrIOp(simd, tail))


@_thread_slices
def complex_profiles(put, meta):
    """Complex add's AVX, compiler-vectorized and scalar C10 loop slots."""
    from xdsl.dialects.builtin import i1
    size, axes, vectorizable = _inner_run(meta)
    position = _position(put, meta, axes)
    def interval(start, end):
        lower = put(arith.ConstantOp(IntegerAttr(start, IndexType())))
        upper = put(arith.ConstantOp(IntegerAttr(end, IndexType())))
        return put(arith.AndIOp(put(arith.CmpiOp(position, lower, 'uge')),
                               put(arith.CmpiOp(position, upper, 'ult'))))
    false = put(arith.ConstantOp(IntegerAttr(0, i1)))
    prefix = size // 8 * 8 if vectorizable else 0
    vector = interval(0, prefix)
    lhs, rhs = scalar_run(meta, 0), rhs_scalar_run(meta)
    if not vectorizable:
        fused = interval(0, size // 2 * 2)
        imag_other = interval(0, size // 4 * 4)
        return vector, fused, fused, false, false, imag_other
    if lhs or rhs:
        start = prefix + 1
        fused = interval(start, start + max(0, size - start) // 4 * 4)
        if lhs:
            true = put(arith.ConstantOp(IntegerAttr(1, i1)))
            return vector, fused, false, false, true, true
        # The last scalar of a four-element RHS-broadcast tail uses a
        # packed two-component add with the product in its first register.
        last = interval(size - 1, size) if size - prefix == 4 else false
        real_first = put(arith.OrIOp(vector, last))
        imag_first = put(arith.OrIOp(real_first, fused))
        true = put(arith.ConstantOp(IntegerAttr(1, i1)))
        return vector, fused, real_first, imag_first, true, true
    fused = interval(0, size // 4 * 4)
    first = put(arith.OrIOp(vector, fused))
    return vector, fused, first, first, false, false
