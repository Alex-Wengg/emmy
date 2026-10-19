"""Kernels that compute the same values from the same buffers are merged into one, before any schedule."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Assign, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.pipeline import Pipeline
from tests.compiler.terms import projection, slab

_M, _N = Axis("m", 8), Axis("n", 16)


def _tile(name: str, reads: tuple, body: tuple, value: str) -> TileOp:
    return TileOp(
        op=projection(reads, body, (value,)),
        name=name,
        place=Placement(free=(_M, _N)),
        axes=(_M, _N),
        output_specs=(OutputSpec(Write(name, (Var("m"), Var("n")), value)),),
    )


def _program(*, second_reads: str = "x", outputs: tuple[str, ...] = ("z",)) -> Graph:
    """``z = -x + -x`` with each ``-x`` its own kernel: what a cut leaves when fusion computed one value in two cones."""
    graph = Graph()
    for name in ("x", "w"):
        graph.add_node(InputOp(), [], Tensor(name, (8, 16), "f32"), node_id=name)
    for name, source in (("y1", "x"), ("y2", second_reads)):
        tile = _tile(name, (slab("v", source, "m", "n"),), (Assign(name="r", op="negative", args=("v",)),), "r")
        graph.add_node(tile, [source], Tensor(name, (8, 16), "f32"), node_id=name)
    total = _tile("z", (slab("a", "y1", "m", "n"), slab("b", "y2", "m", "n")), (Assign(name="s", op="add", args=("a", "b")),), "s")
    graph.add_node(total, ["y1", "y2"], Tensor("z", (8, 16), "f32"), node_id="z")
    graph.inputs, graph.outputs = ["x", "w"], list(outputs)
    return graph


def _tiles(graph: Graph) -> dict[str, TileOp]:
    return {node_id: node.op for node_id, node in graph.nodes.items() if isinstance(node.op, TileOp)}


def test_two_kernels_with_one_computation_become_one() -> None:
    graph = Pipeline.build(["tile/merge"]).run(_program())
    graph.validate()
    [kept] = sorted(set(_tiles(graph)) - {"z"})
    assert len(_tiles(graph)) == 2 and graph.nodes["z"].inputs == [kept, kept]


def test_a_kernel_that_is_a_program_output_keeps_its_name() -> None:
    graph = Pipeline.build(["tile/merge"]).run(_program(outputs=("z", "y1")))
    graph.validate()
    assert sorted(_tiles(graph)) == ["y1", "z"] and graph.outputs == ["z", "y1"]
    assert graph.nodes["z"].inputs == ["y1", "y1"]


def test_kernels_over_different_buffers_stay_two() -> None:
    graph = Pipeline.build(["tile/merge"]).run(_program(second_reads="w"))
    assert sorted(_tiles(graph)) == ["y1", "y2", "z"]
