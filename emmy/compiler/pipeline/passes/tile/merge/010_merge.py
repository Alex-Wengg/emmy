"""Merge a kernel into another that computes the same values from the same buffers.

Loop fusion is maximal, so a producer with several readers is computed inside each reader's cone, and the cut pass
can carve that value out of every one of them: kernels with one body over one set of buffers. The kernel set is
decided by then, so this keeps one of them and points the others' readers at its output — the duplicated work a cut
exposes, removed before any schedule is composed. It decides nothing: a merge has one outcome, and the kernels it
keeps are the ones the cut pass already minted."""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, _rename_buf_in_op
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def _mergeable(node: Node) -> bool:
    # ponytail: one output only, the shape every duplicate a cut mints has had; a multi-output merge needs the
    # splice's output map.
    op = node.op
    return isinstance(op, TileOp) and op.op is not None and not op.carries and len(node.outputs) == 1 and node.outputs[0].alias is None


def _computation(node: Node):
    """``node``'s kernel body with its output named by position, and that output's shape and dtype."""
    (tensor,) = node.outputs
    return _rename_buf_in_op(node.op, node.id, "__merged").loop_body, tensor.shape, tensor.dtype


def rewrite(match: Match, root: Node) -> Graph:
    graph = match.graph
    if not _mergeable(root):
        raise RuleSkipped("not a single-output stateless tile kernel")
    identity = root.op.identity_key(structural=False, with_io=True)
    for node in graph.nodes.values():
        # The kept kernel takes the merged one's name, so it must not be a program output itself.
        if node is root or node.id in graph.outputs or not _mergeable(node):
            continue
        if node.op.identity_key(structural=False, with_io=True) == identity and _computation(node) == _computation(root):
            fragment = Graph()
            fragment.add_node(InputOp(), [], node.outputs[0], node_id=node.id)
            fragment.outputs = [node.id]
            return fragment
    raise RuleSkipped("no other kernel computes this one's values")
