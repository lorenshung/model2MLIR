"""ATen CPU MT19937 and scalar distributions, emitted as standard MLIR.

State contains 624 unsigned words (represented in i64) and the next cursor.
The initialized engine state is seed-derived; no distribution samples are
computed during import. Twists, tempering and sampling all execute at runtime.
"""

from math import prod

from xdsl.dialects import arith, math, scf, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    SymbolRefAttr,
    TensorType,
    UnregisteredOp,
    i64,
)
from xdsl.ir import Block, Region

F64 = Float64Type()
STATE = TensorType(i64, [625])
CACHE = TensorType(F64, [2])


def initial_state(seed):
    words = [seed & 0xFFFFFFFF]
    for j in range(1, 624):
        words.append((1812433253 * (words[-1] ^ (words[-1] >> 30)) + j) & 0xFFFFFFFF)
    return words + [624]


class Emitter:
    def __init__(self, ops=None, state=None):
        self.ops = [] if ops is None else ops
        self.state = state

    def emit(self, op):
        self.ops.append(op)
        return op.results[0] if op.results else None

    def integer(self, value, typ=i64):
        return self.emit(arith.ConstantOp(IntegerAttr(value, typ), typ))

    def floating(self, value, typ=F64):
        return self.emit(arith.ConstantOp(FloatAttr(value, typ), typ))

    def binary(self, cls, left, right):
        return self.emit(cls(left, right))

    def load(self, name, typ):
        return self.emit(
            UnregisteredOp.with_name("ml_program.global_load").create(
                result_types=[typ], properties={"global": SymbolRefAttr(name)}
            )
        )

    def store(self, name, value):
        self.emit(
            UnregisteredOp.with_name("ml_program.global_store").create(
                operands=[value], properties={"global": SymbolRefAttr(name)}
            )
        )

    def extract(self, value, coordinate):
        return self.emit(tensor.ExtractOp(value, [coordinate], value.type.element_type))

    def insert(self, value, array, coordinate):
        return self.emit(tensor.InsertOp(value, array, [coordinate]))

    def loop(self, count, values, fn):
        lb, ub, step = [self.integer(x, IndexType()) for x in (0, count, 1)]
        block = Block(arg_types=[IndexType(), *[v.type for v in values]])
        child = Emitter()
        outputs = fn(child, *block.args)
        child.ops.append(scf.YieldOp(*outputs))
        block.add_ops(child.ops)
        op = scf.ForOp(lb, ub, step, values, Region(block))
        self.ops.append(op)
        return list(op.results)

    def choose(self, cond, types, yes, no):
        blocks = []
        for fn in (yes, no):
            child = Emitter(state=self.state)
            outputs = fn(child)
            block = Block()
            block.add_ops([*child.ops, scf.YieldOp(*outputs)])
            blocks.append(Region(block))
        op = scf.IfOp(cond, types, *blocks)
        self.ops.append(op)
        return list(op.results)

    def word(self):
        """Next uint32, including ATen's in-place (not parallel) twist."""
        cursor_coord = self.integer(624, IndexType())
        cursor = self.extract(self.state, cursor_coord)
        needs_twist = self.emit(arith.CmpiOp(cursor, self.integer(624), "eq"))

        def twist(e):
            def body(b, j, state):
                one = b.integer(1, IndexType())
                following = b.binary(
                    arith.RemUIOp,
                    b.binary(arith.AddiOp, j, one),
                    b.integer(624, IndexType()),
                )
                distant = b.binary(
                    arith.RemUIOp,
                    b.binary(arith.AddiOp, j, b.integer(397, IndexType())),
                    b.integer(624, IndexType()),
                )
                u, v, w = [b.extract(state, c) for c in (j, following, distant)]
                high = b.binary(arith.AndIOp, u, b.integer(0x80000000))
                low = b.binary(arith.AndIOp, v, b.integer(0x7FFFFFFF))
                mix = b.binary(arith.OrIOp, high, low)
                shifted = b.binary(arith.ShRUIOp, mix, b.integer(1))
                odd = b.binary(arith.AndIOp, v, b.integer(1))
                odd = b.emit(arith.CmpiOp(odd, b.integer(0), "ne"))
                matrix = b.emit(
                    arith.SelectOp(odd, b.integer(0x9908B0DF), b.integer(0))
                )
                word = b.binary(
                    arith.XOrIOp, w, b.binary(arith.XOrIOp, shifted, matrix)
                )
                return [b.insert(word, state, j)]

            state = e.loop(624, [self.state], body)[0]
            return [e.insert(e.integer(0), state, cursor_coord)]

        self.state = self.choose(needs_twist, [STATE], twist, lambda e: [self.state])[0]
        cursor = self.extract(self.state, cursor_coord)
        coord = self.emit(arith.IndexCastOp(cursor, IndexType()))
        y = self.extract(self.state, coord)
        self.state = self.insert(
            self.binary(arith.AddiOp, cursor, self.integer(1)), self.state, cursor_coord
        )
        for shift, mask, left in [
            (11, None, False),
            (7, 0x9D2C5680, True),
            (15, 0xEFC60000, True),
            (18, None, False),
        ]:
            part = self.binary(
                arith.ShLIOp if left else arith.ShRUIOp, y, self.integer(shift)
            )
            if mask is not None:
                part = self.binary(arith.AndIOp, part, self.integer(mask))
            y = self.binary(arith.XOrIOp, y, part)
        return y

    def uniform(self, typ):
        bits = 53 if typ == F64 else 24
        value = self.word()
        if bits == 53:
            value = self.binary(
                arith.OrIOp,
                self.binary(arith.ShLIOp, value, self.integer(32)),
                self.word(),
            )
        value = self.binary(arith.AndIOp, value, self.integer((1 << bits) - 1))
        value = self.emit(arith.UIToFPOp(value, typ))
        return self.binary(arith.MulfOp, value, self.floating(2.0**-bits, typ))


def uniform_tensor(result_type):
    e = Emitter()
    state = e.load("global_seed", STATE)
    shape = result_type.get_shape()
    elem = result_type.element_type
    # Half/bfloat16 uniforms use float opmath before conversion.
    from xdsl.dialects.builtin import Float32Type

    sample_type = F64 if elem == F64 else Float32Type()
    empty = e.emit(tensor.EmptyOp([], result_type))

    def body(b, j, state, output):
        b.state = state
        value = b.uniform(sample_type)
        if elem != sample_type:
            value = b.emit(arith.TruncFOp(value, elem))
        coords = []
        for axis, extent in enumerate(shape):
            stride = prod(shape[axis + 1 :])
            c = b.binary(arith.DivUIOp, j, b.integer(stride, IndexType()))
            coords.append(b.binary(arith.RemUIOp, c, b.integer(extent, IndexType())))
        output = b.emit(tensor.InsertOp(value, output, coords))
        return [b.state, output]

    state, output = e.loop(prod(shape), [state, empty], body)
    e.store("global_seed", state)
    return e.ops, output


def normal_tensor(result_type, *, std=1.):
    """ATen scalar cache or contiguous 16-wide Box-Muller fill."""
    shape = result_type.get_shape()
    if prod(shape) >= 16:
        return _normal_vector_tensor(result_type, std=std)
    e = Emitter()
    state = e.load("global_seed", STATE)
    cache = e.load("global_normal_cache", CACHE)
    zero, one = e.integer(0, IndexType()), e.integer(1, IndexType())
    cached_valid, cached_sample = e.extract(cache, zero), e.extract(cache, one)
    empty = e.emit(tensor.EmptyOp([], result_type))
    elem = result_type.element_type

    def body(b, j, state, cached_valid, cached_sample, output):
        b.state = state
        valid = b.emit(arith.CmpfOp(cached_valid, b.floating(0.0), "one"))

        def cached(c):
            return [state, c.floating(0.0), cached_sample, cached_sample]

        def fresh(c):
            u1, u2 = c.uniform(F64), c.uniform(F64)
            negative = c.binary(arith.MulfOp, u2, c.floating(-1.0))
            logged = c.emit(math.Log1pOp(negative))
            radius = c.emit(
                math.SqrtOp(c.binary(arith.MulfOp, c.floating(-2.0), logged))
            )
            angle = c.binary(arith.MulfOp, c.floating(6.283185307179586), u1)
            sample = c.binary(arith.MulfOp, radius, c.emit(math.CosOp(angle)))
            spare = c.binary(arith.MulfOp, radius, c.emit(math.SinOp(angle)))
            return [c.state, c.floating(1.0), spare, sample]

        state, cached_valid, cached_sample, value = b.choose(
            valid, [STATE, F64, F64, F64], cached, fresh
        )
        if std != 1.:
            value = b.binary(arith.MulfOp, value, b.floating(std))
        if elem != F64:
            value = b.emit(arith.TruncFOp(value, elem))
        coords = []
        for axis, extent in enumerate(shape):
            c = b.binary(
                arith.DivUIOp, j, b.integer(prod(shape[axis + 1 :]), IndexType())
            )
            coords.append(b.binary(arith.RemUIOp, c, b.integer(extent, IndexType())))
        output = b.emit(tensor.InsertOp(value, output, coords))
        return [state, cached_valid, cached_sample, output]

    state, cached_valid, cached_sample, output = e.loop(
        prod(shape), [state, cached_valid, cached_sample, empty], body
    )
    cache = e.insert(cached_sample, e.insert(cached_valid, cache, zero), one)
    e.store("global_seed", state)
    e.store("global_normal_cache", cache)
    return e.ops, output


def _normal_vector_tensor(result_type, *, std):
    """DistributionTemplates.h normal_fill, including overlapping tail redraw.

    The 16-wide transform does not use or update the scalar normal cache.
    Half formats fill float buffers; float/double first fill the whole output.
    """
    from xdsl.dialects.builtin import Float32Type, Float16Type, BFloat16Type
    shape, elem = result_type.get_shape(), result_type.element_type
    count = prod(shape)
    sample_type = F64 if elem == F64 else Float32Type()
    tail = count % 16 != 0
    blocks = count // 16 + int(tail)
    half = isinstance(elem, (Float16Type, BFloat16Type))
    draw_count = blocks * 16 if half else count + 16 * int(tail)
    ops, uniforms = uniform_tensor(TensorType(sample_type, [draw_count]))
    e = Emitter(ops)
    empty = e.emit(tensor.EmptyOp([], result_type))

    def outer(b, block_index, output):
        start = b.binary(arith.MuliOp, block_index, b.integer(16, IndexType()))
        last = b.emit(arith.CmpiOp(block_index, b.integer(count // 16, IndexType()), 'eq'))
        output_start = b.emit(arith.SelectOp(last, b.integer(count - 16, IndexType()), start))
        uniform_start = start if half else b.emit(arith.SelectOp(last, b.integer(count, IndexType()), start))
        def pair(c, j, output):
            first = c.binary(arith.AddiOp, uniform_start, j)
            second = c.binary(arith.AddiOp, first, c.integer(8, IndexType()))
            u1, u2 = c.extract(uniforms, first), c.extract(uniforms, second)
            u1 = c.binary(arith.SubfOp, c.floating(1., sample_type), u1)
            radius = c.emit(math.SqrtOp(c.binary(arith.MulfOp, c.floating(-2., sample_type), c.emit(math.LogOp(u1)))))
            theta = c.binary(arith.MulfOp, c.floating(6.283185307179586, sample_type), u2)
            for component, trig in enumerate((math.CosOp, math.SinOp)):
                value = c.binary(arith.MulfOp, radius, c.emit(trig(theta)))
                if std != 1.:
                    value = c.binary(arith.MulfOp, value, c.floating(std, sample_type))
                if elem != sample_type:
                    value = c.emit(arith.TruncFOp(value, elem))
                position = c.binary(arith.AddiOp, output_start, j)
                if component:
                    position = c.binary(arith.AddiOp, position, c.integer(8, IndexType()))
                coords = []
                for axis, extent in enumerate(shape):
                    q = c.binary(arith.DivUIOp, position, c.integer(prod(shape[axis + 1:]), IndexType()))
                    coords.append(c.binary(arith.RemUIOp, q, c.integer(extent, IndexType())))
                output = c.emit(tensor.InsertOp(value, output, coords))
            return [output]
        return b.loop(8, [output], pair)
    output, = e.loop(blocks, [empty], outer)
    return e.ops, output


def permutation_tensor(result_type):
    """ATen small-n forward Fisher-Yates; one uint32 modulo per swap."""
    e = Emitter()
    n = result_type.get_shape()[0]
    state = e.load("global_seed", STATE)
    iota = e.emit(
        arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(result_type, list(range(n))), result_type
        )
    )

    def body(b, j, state, output):
        b.state = state
        remaining = b.binary(
            arith.SubiOp, b.integer(n), b.emit(arith.IndexCastOp(j, i64))
        )
        offset = b.binary(arith.RemUIOp, b.word(), remaining)
        other = b.binary(
            arith.AddiOp, j, b.emit(arith.IndexCastOp(offset, IndexType()))
        )
        x, y = b.extract(output, j), b.extract(output, other)
        output = b.insert(x, b.insert(y, output, j), other)
        return [b.state, output]

    state, output = e.loop(max(0, n - 1), [state, iota], body)
    e.store("global_seed", state)
    return e.ops, output
