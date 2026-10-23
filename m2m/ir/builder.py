"""The one scalar IR builder: ``put(op)`` (or ``b(op)``) emits an op and returns its result."""

import copy

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import FloatAttr, IndexType, IntegerAttr
from xdsl.ir import Block, Region


class Builder:
    """Appends ops to ``block``, or collects them in ``ops`` when there is no block."""

    state = None  # domain state (the RNG's MT state); branches inherit it, loop bodies carry it as a value

    def __init__(self, block=None, elem=None, ops=None):
        self.block, self.elem, self.ops = block, elem, [] if ops is None else ops

    def put(self, op):
        if self.block is None:
            self.ops.append(op)
        else:
            self.block.add_op(op)
        return op.results[0] if op.results else None

    __call__ = put

    def ix(self, n):
        return self.put(arith.ConstantOp(IntegerAttr(n, IndexType())))

    def fp(self, n):
        return self.put(arith.ConstantOp(FloatAttr(float(n), self.elem)))

    def add(self, a, b): return self.put(arith.AddfOp(a, b))
    def sub(self, a, b): return self.put(arith.SubfOp(a, b))
    def mul(self, a, b): return self.put(arith.MulfOp(a, b))
    def div(self, a, b): return self.put(arith.DivfOp(a, b))
    def iadd(self, a, b): return self.put(arith.AddiOp(a, b))
    def imul(self, a, b): return self.put(arith.MuliOp(a, b))

    def cast(self, value, elem=None):
        """PyTorch scalar dtype conversion to ``elem`` (default: the builder's)."""
        from .decompositions import _cast_scalar_arg
        ops, value = _cast_scalar_arg(value, elem or self.elem)
        for op in ops:
            self.put(op)
        return value

    def child(self, block, state=None):
        """A builder of the same kind that emits into ``block`` with ``state``."""
        nested = copy.copy(self)
        nested.block, nested.ops, nested.state = block, [], state
        return nested

    def loop(self, start, end, step, initial, build):
        bounds = [self.ix(n) for n in (start, end, step)]
        body = Block(arg_types=[IndexType(), *(v.type for v in initial)])
        body.add_op(scf.YieldOp(*build(self.child(body), *body.args)))
        loop = scf.ForOp(*bounds, initial, Region(body))
        self.put(loop)
        return list(loop.results)

    def choose(self, condition, types, yes, no):
        regions = []
        for build in (yes, no):
            block = Block()
            block.add_op(scf.YieldOp(*build(self.child(block, self.state))))
            regions.append(Region(block))
        op = scf.IfOp(condition, types, *regions)
        self.put(op)
        return list(op.results)
