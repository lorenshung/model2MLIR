"""CPU pointwise SIMD runs and scalar-tail storage rounding."""


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
        ops = []
        def put(op):
            ops.append(op)
            return op.results[0]
        wide = put(arith.ExtFOp(args[0], f32))
        root = put(SqrtOp(wide))
        if prefix < size:
            rounded = put(arith.TruncFOp(root, elem))
            scalar_root = put(arith.ExtFOp(rounded, f32))
            if prefix:
                position = put(arith.ConstantOp(IntegerAttr(0, IndexType())))
                factor = 1
                for axis in axes:
                    coordinate = put(linalg.IndexOp(axis))
                    stride = put(arith.ConstantOp(IntegerAttr(factor, IndexType())))
                    term = put(arith.MuliOp(coordinate, stride))
                    position = put(arith.AddiOp(position, term))
                    factor *= shape[axis]
                boundary = put(arith.ConstantOp(IntegerAttr(prefix, IndexType())))
                simd = put(arith.CmpiOp(position, boundary, "ult"))
                root = put(arith.SelectOp(simd, root, scalar_root))
            else:
                root = scalar_root
        one = put(arith.ConstantOp(FloatAttr(1., f32)))
        inverse = put(arith.DivfOp(one, root))
        result = put(arith.TruncFOp(inverse, elem))
        return ops, result

    return _elementwise([x], TensorType(elem, shape), build)
