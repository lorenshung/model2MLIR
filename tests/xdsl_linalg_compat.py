"""Standard linalg interpreter support missing from the pinned xDSL release.

Keep source CPU lowerings intact: linalg.index reads the current structured
loop coordinates, including inside nested SCF regions. Each tensor result has
its own destination and indexing map; nested structured loops restore the
outer coordinates on return.
"""
from itertools import product

from xdsl.dialects import linalg
from xdsl.interpreter import impl, register_impls
from xdsl.interpreters.linalg import LinalgFunctions as _LinalgFunctions
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class LinalgFunctions(_LinalgFunctions):
    @impl(linalg.GenericOp)
    def run_generic(self, interpreter, op, args):
        if op.library_call is not None:
            raise NotImplementedError('linalg.generic library_call')
        inputs = args[:len(op.inputs)]
        destinations = args[len(op.inputs):]
        if any(not isinstance(value, ShapedArray) for value in destinations):
            raise NotImplementedError('linalg destinations must be shaped arrays')
        outputs = tuple(value.copy() for value in destinations) if op.results else destinations
        maps = tuple(value.data for value in op.get_indexing_maps())
        ranges = op.get_static_loop_ranges()
        previous = getattr(self, '_coordinates', None)
        try:
            for coordinates in product(*(range(extent) for extent in ranges)):
                self._coordinates = coordinates
                values = tuple(value.load(mapping.eval(coordinates, ()))
                               if isinstance(value, ShapedArray) else value
                               for value, mapping in zip((*inputs, *outputs), maps, strict=True))
                results = interpreter.run_ssacfg_region(op.body, values, 'linalg_loop')
                for output, mapping, result in zip(outputs, maps[len(inputs):], results, strict=True):
                    output.store(mapping.eval(coordinates, ()), result)
        finally:
            self._coordinates = previous
        return outputs if op.results else ()

    @impl(linalg.IndexOp)
    def run_index(self, interpreter, op, args):
        coordinates = getattr(self, '_coordinates', None)
        if coordinates is None:
            raise ValueError('linalg.index outside a structured loop')
        return (coordinates[op.dim.value.data],)
