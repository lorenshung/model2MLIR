"""Exercise loop coordinates and independent destinations through actual IR."""
import numpy as np
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from tests.xdsl_linalg_compat import LinalgFunctions


def test_nested_index_context_and_multiple_output_maps():
    first = builtin.TensorType(builtin.i64, [2, 3])
    second = builtin.TensorType(builtin.i64, [3, 2])
    entry = Block()
    outputs = [tensor.EmptyOp([], first), tensor.EmptyOp([], second)]
    entry.add_ops(outputs)
    body = Block(arg_types=[builtin.i64, builtin.i64])
    inner_type = builtin.TensorType(builtin.i64, [4])
    empty = tensor.EmptyOp([], inner_type)
    inner_body = Block(arg_types=[builtin.i64])
    inner_index = linalg.IndexOp(0)
    inner_value = arith.IndexCastOp(inner_index.result, builtin.i64)
    inner_body.add_ops([inner_index, inner_value, linalg.YieldOp(inner_value.result)])
    inner = linalg.GenericOp(inputs=[], outputs=[empty.results[0]], body=Region(inner_body),
        indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(1))],
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)], result_types=[inner_type])
    body.add_ops([empty, inner])
    i, j = linalg.IndexOp(0), linalg.IndexOp(1)
    a, b = arith.IndexCastOp(i.result, builtin.i64), arith.IndexCastOp(j.result, builtin.i64)
    ten = arith.ConstantOp(builtin.IntegerAttr(10, builtin.i64))
    product = arith.MuliOp(a.result, ten.result)
    total = arith.AddiOp(product.result, b.result)
    body.add_ops([i, j, a, b, ten, product, total, linalg.YieldOp(total.result, a.result)])
    outer = linalg.GenericOp(inputs=[], outputs=[value.results[0] for value in outputs], body=Region(body),
        indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(2)),
                       builtin.AffineMapAttr(AffineMap.from_callable(lambda i, j: (j, i)))],
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2,
        result_types=[first, second])
    entry.add_ops([outer, func.ReturnOp(*outer.results)])
    module = builtin.ModuleOp([func.FuncOp('f', ([], [first, second]), Region(entry))])
    module.verify()
    interpreter = Interpreter(module)
    for implementations in [ArithFunctions(), FuncFunctions(), TensorFunctions(), LinalgFunctions()]:
        interpreter.register_implementations(implementations)
    x, y = interpreter.call_op('f', ())
    np.testing.assert_array_equal(np.array(x.data).reshape(x.shape), [[0, 1, 2], [10, 11, 12]])
    np.testing.assert_array_equal(np.array(y.data).reshape(y.shape), [[0, 1], [0, 1], [0, 1]])
