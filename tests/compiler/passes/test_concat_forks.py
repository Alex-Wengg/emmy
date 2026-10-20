"""Sibling weights read as one constant: the contractions that differ only in their weight become one contraction over
the weights side by side, and compute the same values as the weights read apart."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from emmy.compiler.backend.numpy import NumpyBackend
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.fork import DeferredFork, parallel_expand
from emmy.compiler.pipeline.passes.tile._layout import CONCAT, concat_name, settled_layouts
from emmy.compiler.pipeline.pipeline import Run, _is_structural_option
from emmy.compiler.pipeline.search.pins import spelled_arm

# One input, four weights of three widths; the output reads all of them, so no contraction owns an output.
_CODE = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(32, 24, bias=False)
        self.z = nn.Linear(32, 16, bias=False)
        self.a = nn.Linear(32, 4, bias=False)
        self.b = nn.Linear(32, 4, bias=False)
    def forward(self, x):
        qkv, z, a, b = self.qkv(x), self.z(x), self.a(x), self.b(x)
        s = (a * b).sum(-1, keepdim=True)
        return qkv[..., :16] * z + s * qkv[..., 16:].sum(-1, keepdim=True)
m = M()
m(torch.randn(1, 32))
"""


@functools.cache
def _trace():
    from emmy.commands.trace import graph_from_code

    graph, _, (module, args, _) = graph_from_code(_CODE)
    return graph, module, args


def _traced():
    """The traced program (a fresh copy) beside the one module and inputs it was traced from."""
    graph, module, args = _trace()
    return graph.copy(), module, args


def _lower(joined: bool, *, cut: bool = False) -> tuple[Graph, list[list[dict]]]:
    """The traced program through the cut pass with its weights joined or apart; ``cut`` takes, at each kernel's
    placement fork, the arm that cuts the most seams (the contractions' own kernels), else every kernel stays fused."""
    graph, _, _ = _traced()
    offered: list[list[dict]] = []

    def decide(point):
        knobs = [dict(option.knobs) for option in point.options]
        if any(CONCAT in row.values() for row in knobs):
            offered.append(knobs)
            parallel_expand([option for option in point.options if _is_structural_option(option)], workers=1)
            return next(option for option in point.options if (CONCAT in option.knobs.values()) == joined)
        if cut and any(value == "cut" for row in knobs for value in row.values()):
            return max(point.options, key=lambda option: sum(value == "cut" for value in option.knobs.values()))
        return next((option for option in point.options if not _is_structural_option(option)), point.options[0])

    graph = Pipeline.build(LOOP_PASSES).run(graph)
    graph, _ = Run(Pipeline.build(["tile/lift", "tile/cut"]), Context.from_target((12, 0))).resolve(graph, decide)
    graph.validate()
    return graph, offered


def _run(graph: Graph) -> dict:
    from emmy.compiler.loader.binder import apply_load_ops

    _, module, args = _traced()
    state = {name: tensor.detach().float().numpy() for name, tensor in module.state_dict().items()}
    for node in graph.nodes.values():
        if isinstance(node.op, TileOp):
            op = node.op
            node.op = LoopOp(body=op.op.lower(bound=frozenset(), stores=op.output_specs, axes=op.axes))
    inputs = {name: args[i].detach().float().numpy() for i, name in enumerate(graph.inputs)}
    for node_id, node in graph.nodes.items():
        op = node.op
        if not isinstance(op, ConstantOp):
            continue
        if op.value is not None:
            inputs[node_id] = np.array([op.value], dtype=np.float32)
            continue
        paths = [path for path, _ in op.source_parts] or [op.source_path]
        source = np.concatenate([state[path.removeprefix("m.")] for path in paths], axis=0)
        inputs[node_id] = apply_load_ops(source.reshape(op.source_shape), op.load_ops)
    result, _ = NumpyBackend().run(graph, input_data=inputs)
    return result.outputs


def _family(offered: list[list[dict]]) -> list[str]:
    return sorted(key.partition("@")[2] for row in offered[0] for key, value in row.items() if value == CONCAT)


def test_joined_weights_compute_what_eager_computes() -> None:
    joined, offered = _lower(True)
    apart, _ = _lower(False)
    # The fork offers the weights apart and joined, the joined arm spelling every weight of the family.
    assert len(offered[0]) == 2 and {"LAYOUT": "folded"} in offered[0]
    names = _family(offered)
    assert len(names) == 4
    constants = dict(joined.loadable_constants())
    assert concat_name(names) in constants and not set(names) & set(constants)
    assert len(constants[concat_name(names)].source_parts) == 4
    _, module, args = _traced()
    eager = module(*args).detach().float().numpy()
    for graph in (joined, apart):
        (value,) = _run(graph).values()
        np.testing.assert_allclose(value, eager, rtol=1e-5, atol=1e-5)


def test_a_cut_of_joined_weights_keeps_one_workspace_per_weight() -> None:
    """One producer reads the joined weight and stores each weight's columns to a workspace of its own, so each
    weight's readers read what the weights read apart give them. (Not executed here: the loop runner sizes an output
    by its write's index over the whole nest, not under the guard such a workspace is stored under.)"""
    joined, offered = _lower(True, cut=True)
    apart, _ = _lower(False, cut=True)
    names = _family(offered)
    (producer,) = (node for node in joined.nodes.values() if concat_name(names) in node.inputs)

    def widths(nodes) -> list[tuple]:
        return sorted(tuple(dim.as_static() for dim in tensor.shape) for node in nodes for tensor in node.outputs)

    assert widths([producer]) == widths(node for node in apart.nodes.values() if set(names) & set(node.inputs))


def test_a_row_spells_the_joined_arm_only_when_it_joins_every_weight() -> None:
    apart = DeferredFork(lambda: None, {"LAYOUT": "folded"})
    joined = DeferredFork(lambda: None, {"LAYOUT@a": CONCAT, "LAYOUT@b": CONCAT}, structural=True)
    options = [apart, joined]
    assert spelled_arm(options, {})[0] is apart
    assert spelled_arm(options, {"LAYOUT@a": "source", "LAYOUT@b": "source"})[0] is apart
    assert spelled_arm(options, {"LAYOUT@a": CONCAT})[0] is apart
    assert spelled_arm(options, {"LAYOUT@a": CONCAT, "LAYOUT@b": CONCAT})[0] is joined


def _concat_offers(graph: Graph) -> list[list[dict]]:
    """The arms of every concatenation fork a compile of ``graph`` meets, each kernel deciding its weights apart."""
    offered = []

    def decide(point):
        knobs = [dict(option.knobs) for option in point.options]
        if any(CONCAT in row.values() or row == {"LAYOUT": "folded"} for row in knobs):
            offered.append(knobs)
        return next((option for option in point.options if not _is_structural_option(option)), point.options[0])

    Run(Pipeline.build(["tile/lift", "tile/cut"]), Context.from_target((12, 0))).resolve(graph, decide)
    return offered


@pytest.mark.parametrize("settled", [CONCAT, "folded"])
def test_a_family_settled_by_another_twin_is_offered_that_storage_only(settled: str) -> None:
    graph = Pipeline.build(LOOP_PASSES).run(_traced()[0])
    names = sorted(key.partition("@")[2] for row in _concat_offers(graph.copy())[0] for key, value in row.items() if value == CONCAT)
    with settled_layouts({name: settled for name in names}):
        offered = _concat_offers(graph)
    assert offered[0] == ([{f"LAYOUT@{name}": CONCAT for name in names}] if settled == CONCAT else [{"LAYOUT": "folded"}])


def test_a_twin_reading_a_joined_weight_alone_holds_it_twice() -> None:
    from emmy.serving.twins import LayoutConflict, LayoutScope

    scope = LayoutScope()
    scope.record("gdn1", [(concat_name(("w2", "w1")), True)])
    assert scope.settled() == {"cat__w1__w2": "folded", "w1": CONCAT, "w2": CONCAT}
    with pytest.raises(LayoutConflict, match="concat"):
        scope.record("gdn16", [("w1", True)])
    with pytest.raises(LayoutConflict):
        LayoutScope().record("gdn64", [("cat__w1__w2", True), ("w1", True)])
