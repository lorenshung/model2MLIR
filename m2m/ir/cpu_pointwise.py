"""CPU pointwise SIMD runs and scalar-tail storage rounding."""


def linear_position(put, shape, axes):
    """Position of the current linalg element along ``axes``, innermost first."""
    from xdsl.dialects import arith, linalg
    from xdsl.dialects.builtin import IndexType, IntegerAttr
    position, factor = put(arith.ConstantOp(IntegerAttr(0, IndexType()))), 1
    for axis in axes:
        coordinate = put(linalg.IndexOp(axis))
        stride = put(arith.ConstantOp(IntegerAttr(factor, IndexType())))
        position = put(arith.AddiOp(position, put(arith.MuliOp(coordinate, stride))))
        factor *= shape[axis]
    return position


def flat_conversion_prefix(put, output, block_size):
    """Vector slots in a dense CPU conversion, including OpenMP range tails."""
    import torch
    from xdsl.dialects import arith, linalg
    from xdsl.dialects.builtin import IndexType, IntegerAttr
    def constant(value):
        return put(arith.ConstantOp(IntegerAttr(value, IndexType())))
    total = output.numel()
    position = linear_position(put, output.shape, sorted(range(output.ndim), key=lambda a: output.stride(a)))
    tasks = min(torch.get_num_threads(), (total + 32767) // 32768) if total > 32768 else 1
    chunk = max(1, (total + tasks - 1) // tasks)
    local = put(arith.RemUIOp(position, constant(chunk)))
    start = put(arith.SubiOp(position, local))
    remaining = put(arith.SubiOp(constant(total), start))
    length = put(arith.MinUIOp(remaining, constant(chunk)))
    prefix = put(arith.MuliOp(put(arith.DivUIOp(length, constant(block_size))), constant(block_size)))
    return put(arith.CmpiOp(local, prefix, 'ult'))


def half_to_float_copy(value, source):
    """CPU copy's dense Half->float vector conversion and software tail.

    The contiguous vec::convert loop uses F16C for complete float vectors.
    Its remaining scalar conversions emit positive 0x7fffffff for a NaN.
    Strided copies use the AVX2 scalar/partial-vector conversion instead.
    """
    import torch
    from torch._prims_common import is_non_overlapping_and_dense
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from .builder import Builder
    from .decompositions import _elementwise
    if not is_non_overlapping_and_dense(source):
        return [], value
    def build(args, elem):
        put = Builder()
        width = 512 if torch.backends.cpu.get_cpu_capability() == 'AVX512' else 256
        vector = flat_conversion_prefix(put, source, width // 32)
        bits = put(arith.ConstantOp(IntegerAttr(0x7fffffff, i32)))
        nan = put(arith.BitcastOp(bits, elem))
        missing = put(arith.CmpfOp(args[0], args[0], 'uno'))
        tail = put(arith.SelectOp(missing, nan, args[0]))
        return put.ops, put(arith.SelectOp(vector, args[0], tail))
    return _elementwise([value], value.type, build)


def inner_run(source, output):
    """TensorIterator's coalesced inner extent and unit/scalar input stride.

    The output's stride order determines traversal. A broadcast outer axis
    cannot coalesce with an inner source row. Padded inputs use scalar loops.
    """
    axes = sorted(range(output.ndim), key=lambda a: output.stride(a))
    size, step, out_step = 1, None, None
    group = []
    for axis in axes:
        if output.shape[axis] == 1:
            continue
        si, so = source.stride(axis), output.stride(axis)
        if step is None:
            step, out_step = si, so
        elif si != size * step or so != size * out_step:
            break
        group.append(axis)
        size *= output.shape[axis]
    return size, step in (0, 1) and out_step in (None, 1), group


def reduced_rsqrt(x, meta):
    """Native half/BF16 rsqrt: scalar sqrt narrows before division.

    The SIMD kernel instead computes 1/sqrt in float and narrows once.
    Loops.h processes two complete vectors per iteration, then scalar tails.
    """
    import torch
    from xdsl.dialects import arith, linalg
    from xdsl.dialects.math import SqrtOp
    from xdsl.dialects.builtin import FloatAttr, IndexType, IntegerAttr, TensorType, f32
    from .builder import Builder
    from .decompositions import _elementwise, _fx_arg, _shape_of, _t_elem

    shape, elem = _shape_of(x), _t_elem(x)
    arg = _fx_arg(meta, 0)
    source = getattr(arg, "meta", {}).get("val", arg)
    output = meta["val"]
    if source is None:
        source = output
    if source.device.type != "cpu":
        from xdsl.dialects.math import RsqrtOp
        from .decompositions import _cast_tensor, _un_build
        wide_ops, wide = _cast_tensor(x, shape, f32)
        built = _elementwise([wide], TensorType(f32, shape), _un_build(RsqrtOp))
        cast_ops, result = _cast_tensor(built[1], shape, elem)
        return [*wide_ops, *built[0], *cast_ops], result
    size, vectorizable, axes = inner_run(source, output)
    bits = 512 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 256
    block_size = 2 * bits // 16
    prefix = size // block_size * block_size if vectorizable else 0

    def build(args, out_elem):
        put = Builder()
        wide = put(arith.ExtFOp(args[0], f32))
        root = put(SqrtOp(wide))
        if prefix < size:
            rounded = put(arith.TruncFOp(root, elem))
            scalar_root = put(arith.ExtFOp(rounded, f32))
            if prefix:
                position = linear_position(put, shape, axes)
                boundary = put(arith.ConstantOp(IntegerAttr(prefix, IndexType())))
                simd = put(arith.CmpiOp(position, boundary, "ult"))
                root = put(arith.SelectOp(simd, root, scalar_root))
            else:
                root = scalar_root
        one = put(arith.ConstantOp(FloatAttr(1., f32)))
        inverse = put(arith.DivfOp(one, root))
        return put.ops, put(arith.TruncFOp(inverse, elem))

    return _elementwise([x], TensorType(elem, shape), build)
