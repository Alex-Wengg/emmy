"""Offer a constant's folded transpose and its original storage layout as measured alternatives."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace

from emmy.compiler.dim import to_dim
from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import TransposeOp
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.stmt import Load
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import Match
from emmy.compiler.pipeline.fork import DeferredFork
from emmy.compiler.pipeline.knob import family_pins
from emmy.compiler.pipeline.passes.tile._row import io_shapes, reformed
from emmy.compiler.pipeline.passes.tile._split import add_output_piece, redirects
from emmy.compiler.pipeline.passes.tile.cut import takes_kernel_set_decision


def _source_shape(node: Node) -> tuple | None:
    op = node.op
    if not isinstance(op, ConstantOp) or op.value is not None or not op.load_ops or len(node.output.shape) != 2:
        return None
    last = op.load_ops[-1]
    if not isinstance(last, TransposeOp) or len(last.axes) != 2:
        return None
    axes = tuple(axis % 2 for axis in last.axes)
    return tuple(reversed(node.output.shape)) if axes in {(0, 1), (1, 0)} else None


def _edit(term: Fold, names: dict[str, str]) -> Fold:
    return _edit_loads(term, lambda stmt: replace(stmt, input=names[stmt.input], index=tuple(reversed(stmt.index))), set(names))


def _edit_loads(term: Fold, edit, inputs: set[str]) -> Fold:
    def load(stmt):
        return edit(stmt) if isinstance(stmt, Load) and stmt.input in inputs else stmt

    return replace(
        term,
        operands=tuple(_edit_loads(operand, edit, inputs) for operand in term.operands),
        lift=replace(term.lift, body=term.lift.body.map(load)),
    )


#: A kernel reads constant ``name`` in its source layout from buffer ``name + SOURCE_SUFFIX``.
SOURCE_SUFFIX = "__source"
#: The temporary a source-layout kernel's outputs travel under while its fragment is spliced in.
_LAYOUT = "__layout"

_SETTLED: ContextVar[Mapping[str, str] | None] = ContextVar("settled_layouts", default=None)


@contextlib.contextmanager
def settled_layouts(layouts: Mapping[str, str]):
    """Compile with the layout of each constant in ``layouts`` (buffer name -> ``folded`` / ``source``) already
    settled, by another program that shares its device copy: the fork offers a settled constant in that layout only.
    Unlike a ``LAYOUT`` pin, which asks for a choice the compile must realize, this constrains a fork only where one
    arises."""
    token = _SETTLED.set(dict(layouts))
    try:
        yield
    finally:
        _SETTLED.reset(token)


def _source_fragment(graph: Graph, root: Node, names: tuple[str, ...]) -> Graph:
    source = {name: f"{name}{SOURCE_SUFFIX}" for name in names}
    fragment = Graph()
    tile: TileOp = root.op
    shapes = io_shapes(tile)  # every buffer the piece reads, each raw source among them
    for name in root.inputs:
        if name not in source:
            fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
            continue
        folded = graph.producer(name)
        assert folded is not None and isinstance(folded.op, ConstantOp)
        raw = source[name]
        shape = shapes[raw] = _source_shape(folded)
        assert shape is not None
        op = replace(folded.op, load_ops=folded.op.load_ops[:-1])
        if (existing := graph.buffer(raw)) is not None:
            assert existing.shape == shape and graph.producer(raw).op == op
            fragment.add_node(InputOp(), [], existing, node_id=raw)
        else:
            fragment.add_node(op, [], Tensor(raw, shape, folded.output.dtype), node_id=raw)
    piece = reformed(replace(tile, op=_edit(tile.op, source), source=None, layout_decided=(*tile.layout_decided, *names)), shapes)
    inputs = [source.get(name, name) for name in root.inputs]
    return add_output_piece(fragment, root, piece, inputs, suffix=_LAYOUT)


def _loads(tile: TileOp) -> dict[str, list[Load]]:
    loads: dict[str, list[Load]] = {}
    if tile.loop_body is not None:
        for stmt in tile.loop_body.iter():
            if isinstance(stmt, Load):
                loads.setdefault(stmt.input, []).append(stmt)
    return loads


def _undecided(op, name: str) -> bool:
    """Whether this kernel still offers a layout fork for constant ``name``. It does not when it takes no kernel-set
    decision any more (:func:`takes_kernel_set_decision`) or has no loop body yet, when it carries a state (its one
    kernel-set decision is the split), when its layout of ``name`` is already decided, or when it reads ``name``
    other than through 2-D indices (or not at all)."""
    if not takes_kernel_set_decision(op) or op.loop_body is None or op.carries or name in op.layout_decided:
        return False
    reads = _loads(op).get(name)
    return bool(reads) and all(len(load.index) == 2 for load in reads)


def fork_less_reads(graph: Graph) -> set[str]:
    """The transposed constants of ``graph`` that some kernel reads with no layout fork (:func:`_undecided`): those
    stay folded whatever layout another reader decides, so a program sharing them settles them folded."""
    return {
        name
        for node in graph.nodes.values()
        for name in node.inputs
        if (producer := graph.producer(name)) is not None and _source_shape(producer) is not None and not _undecided(node.op, name)
    }


def layout_readers(graph: Graph, root: Node, names: tuple[str, ...]) -> list[Node]:
    """Every other kernel of ``graph`` still offered the layout fork of a constant in ``names`` (:func:`_undecided`):
    the readers a decision at ``root`` settles with it (:func:`_taken_layout`), whose rows price that decision beside
    ``root``'s own — the layout is the program's, so its price is the program's."""
    return [
        node
        for node in graph.nodes.values()
        if node.id != root.id and any(name in node.inputs and _undecided(node.op, name) for name in names)
    ]


def reader_kernel(graph: Graph, node: Node, source: tuple[str, ...]) -> TileOp:
    """``node``'s kernel as a layout decision leaves it: itself when ``source`` names none of its constants, else the
    kernel that reads those in their source layout — what the arm's rows are looked up on for this reader."""
    names = tuple(name for name in source if name in node.inputs)
    if not names:
        return node.op.with_io(graph, node)
    fragment = _source_fragment(graph, node, names)
    piece = next(n for n in fragment.nodes.values() if isinstance(n.op, TileOp))
    return piece.op.with_io(fragment, piece)


def _taken_layout(graph: Graph, root: Node, name: str) -> str | None:
    """The layout another kernel of this program, or a program settling it (:func:`settled_layouts`), already reads
    constant ``name`` in, or ``None``.

    A reader the cut pass no longer offers a layout fork for ``name`` (see :func:`_undecided`) reads it folded for good,
    whatever another program settled: serving binds one device copy per layout, so a second layout would hold the
    weight twice, and a program that cannot take a settled layout says so where its layouts are recorded."""
    if graph.buffer_users(f"{name}{SOURCE_SUFFIX}"):
        return "source"
    if any(user != root.id and not _undecided(graph.nodes[user].op, name) for user in graph.buffer_users(name)):
        return "folded"
    return (_SETTLED.get() or {}).get(name)


def layout_forks(match: Match, root: Node) -> list[DeferredFork] | None:
    """The first undecided transposed constant read by this kernel, with a joint arm for equal reads.

    Equal read expressions can be two channels of one contraction. Offering their joint source arm
    lets evidence compare that kernel with the folded one without requiring either mixed layout to win first.
    A constant another kernel of the program already reads in one layout is offered in that layout only.
    """
    tile: TileOp = root.op
    if tile.op is None or tile.loop_body is None:
        return None
    loads = _loads(tile)
    eligible = [
        name
        for name in root.inputs
        if (node := match.graph.producer(name)) is not None and _source_shape(node) is not None and _undecided(tile, name)
    ]
    if not eligible:
        return None
    first = eligible[0]
    taken = _taken_layout(match.graph, root, first)
    signature = tuple(load.index for load in loads[first])
    # The group is the equal reads in the same state as the first: all undecided, or all taken the same way, so a
    # group a program or a record settled source is still offered as the one joint decision it was taken as.
    group = tuple(
        name
        for name in eligible
        if tuple(load.index for load in loads[name]) == signature and _taken_layout(match.graph, root, name) == taken
    )
    folded = DeferredFork(lambda: replace(tile, layout_decided=(*tile.layout_decided, first)), {f"LAYOUT@{first}": "folded"})
    output = redirects(root, _LAYOUT)
    graph = match.graph
    single = DeferredFork(lambda: _source_fragment(graph, root, (first,)), {f"LAYOUT@{first}": "source"}, structural=True, output=output)
    joint = (
        DeferredFork(
            lambda: _source_fragment(graph, root, group),
            {f"LAYOUT@{name}": "source" for name in group},
            structural=True,
            output=output,
        )
        if len(group) > 1
        else None
    )
    if taken is not None:
        return [folded if taken == "folded" else joint or single]
    options = [folded, single, *([joint] if joint is not None else [])]
    pins = dict(family_pins("LAYOUT"))
    if any(value not in {"folded", "source", CONCAT} for value in pins.values()):
        raise ValueError("LAYOUT must be 'folded', 'source' or 'concat'")
    pins = {key: value for key, value in pins.items() if value != CONCAT}
    if joint is not None and all(pins.get(f"LAYOUT@{name}", pins.get("LAYOUT")) == "source" for name in group):
        return [joint]
    return [option for option in options if all(pins.get(key, pins.get("LAYOUT", value)) == value for key, value in option.knobs.items())]


#: The ``LAYOUT`` value that stores sibling weights as one constant, concatenated along their output features.
CONCAT = "concat"
#: The concatenated constant's buffer is its first part's name with this suffix.
CONCAT_SUFFIX = "__cat"
_CONCAT = "__concat"


def _concat_part(graph: Graph, name: str) -> Node | None:
    """The constant ``name`` when it can be one part of a concatenation: a checkpoint tensor read through a load chain
    that ends in a transpose of its two axes, so its source rows are its output features."""
    node = graph.producer(name)
    if node is None or _source_shape(node) is None or node.op.source_path is None or node.op.source_graph is not None:
        return None
    return node


def _concat_fragment(graph: Graph, root: Node, names: tuple[str, ...]) -> Graph:
    """``root``'s kernel reading the constants ``names`` as one constant: their source tensors concatenated along
    the output features (``ConstantOp.source_parts``) and each read at its offset there."""
    parts = [graph.producer(name) for name in names]
    k = parts[0].output.shape[0].as_static()
    widths = [part.output.shape[1].as_static() for part in parts]
    offsets = dict(zip(names, (sum(widths[:i]) for i in range(len(widths))), strict=True))
    width = sum(widths)
    cat = f"{names[0]}{CONCAT_SUFFIX}"
    op = replace(
        parts[0].op,
        source_path=None,
        source_parts=tuple((part.op.source_path, tuple(part.op.source_shape)) for part in parts),
        source_shape=(width, k),
    )
    fragment = Graph()
    tile: TileOp = root.op
    shapes = io_shapes(tile)
    shapes[cat] = (to_dim(k), to_dim(width))
    for name in root.inputs:
        if name not in offsets:
            fragment.add_node(InputOp(), [], graph.buffer(name), node_id=name)
    if (existing := graph.buffer(cat)) is not None:
        fragment.add_node(InputOp(), [], existing, node_id=cat)
    else:
        fragment.add_node(op, [], Tensor(cat, (to_dim(k), to_dim(width)), parts[0].output.dtype), node_id=cat)

    def read(load: Load) -> Load:
        row, column = load.index
        offset = offsets[load.input]
        return replace(load, input=cat, index=(row, BinaryExpr("+", column, Literal(offset, "int")) if offset else column))

    piece = reformed(replace(tile, op=_edit_loads(tile.op, read, set(names)), source=None), shapes)
    inputs = list(dict.fromkeys(cat if name in offsets else name for name in root.inputs))
    return add_output_piece(fragment, root, piece, inputs, suffix=_CONCAT)


def concat_forks(match: Match, root: Node) -> list[DeferredFork] | None:
    """PROTOTYPE: the concatenation of sibling weights this kernel's contractions read (``_cut.sibling_weights``),
    offered only where ``LAYOUT@<weight>=concat`` pins every weight of the family."""
    from emmy.compiler.pipeline.passes.tile._cut import sibling_weights  # noqa: PLC0415

    tile: TileOp = root.op
    if tile.op is None or tile.loop_body is None or tile.carries:
        return None
    pins = dict(family_pins("LAYOUT"))
    graph = match.graph
    import os  # noqa: PLC0415 — PROTOTYPE switch

    offer = os.environ.get("HFUSE_OFFER") == "1"
    for names in sibling_weights(tile):
        marker = "concat:" + "+".join(names)
        if marker in tile.layout_decided:
            continue
        parts = [_concat_part(graph, name) for name in names]
        if any(part is None for part in parts):
            continue
        if len({(part.output.shape[0], part.output.dtype, part.op.load_ops, part.op.source_dtype) for part in parts}) > 1:
            continue
        concat = DeferredFork(
            lambda names=names: _concat_fragment(graph, root, names),
            {f"LAYOUT@{name}": CONCAT for name in names},
            structural=True,
            output=redirects(root, _CONCAT),
        )
        if all(pins.get(f"LAYOUT@{name}") == CONCAT for name in names):
            return [concat]
        if offer:
            apart = DeferredFork(lambda marker=marker: replace(tile, layout_decided=(*tile.layout_decided, marker)), {"LAYOUT": "folded"})
            return [apart, concat]
    return None
