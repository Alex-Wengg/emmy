"""Where a kernel waits on the grid ahead of it (``griddepcontrol.wait``) and releases the grid behind it
(``griddepcontrol.launch_dependents``): the prologue fills that read only program constants are issued before
the wait, everything else after it."""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import (
    BinaryExpr,
    CpAsyncCommit,
    CpAsyncCopy,
    KernelOp,
    Literal,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    StridedLoop,
    Sync,
    Tile,
    TmaDescriptor,
    TmaLoad,
    Var,
)
from emmy.compiler.ir.kernel.render import render_kernelop
from emmy.compiler.ir.stmt import Body, Cond, Load, Write
from emmy.compiler.tensor import Tensor

_TID0 = BinaryExpr("==", Var("t"), Literal(0))
_TENSORS = {"w": Tensor("w", (64, 64), "f16"), "x": Tensor("x", (64, 64), "f16"), "y": Tensor("y", (64,), "f32")}


def _copy(desc: str, slab: str) -> TmaLoad:
    return TmaLoad(
        smem=slab, smem_index=(Literal(0), Literal(0)), desc=desc, coords=(Literal(0), Literal(0)), mbar="_mbar", mbar_slot=Literal(0)
    )


def _kernel(*prologue) -> KernelOp:
    """A TMA-staged kernel: setup, ``prologue``, then a consumer that waits on the barrier and stores."""
    setup = (
        TmaDescriptor(name="_desc_w", src_buf="w", src_shape=(64, 64), box_extents=(16, 64), dtype="__half"),
        TmaDescriptor(name="_desc_x", src_buf="x", src_shape=(64, 64), box_extents=(16, 64), dtype="__half"),
        Smem(name="_w_smem", extents=(16, 64), dtype="__half", align=128),
        Smem(name="_x_smem", extents=(16, 64), dtype="__half", align=128),
        Smem(name="_mbar", extents=(1,), dtype="unsigned long long"),
        Cond(cond=_TID0, body=(MbarrierInit(mbar="_mbar", count=1, slot=Literal(0)),)),
        Sync(),
    )
    consume = (
        MbarrierWait(mbar="_mbar", phase=Literal(0), slot=Literal(0)),
        Write(output="y", index=(Var("t"),), values=("v",)),
    )
    body = Tile(axes=(Axis("t", 64),), body=Body((*setup, *prologue, *consume)), block_threads=64)
    return KernelOp(body=Body((body,)), name="k_probe")


def _fill(*copies) -> Cond:
    return Cond(cond=_TID0, body=(MbarrierArriveExpectTx(mbar="_mbar", bytes_=4096, slot=Literal(0)), *copies))


def _body(source: str) -> str:
    """The rendered function body: the signature also names every buffer and descriptor."""
    return source[source.index(") {\n", source.index("__global__")) :]


def test_the_constant_half_of_a_fill_lands_before_the_wait_and_the_rest_after() -> None:
    src = _body(
        render_kernelop(
            _kernel(
                Load(name="v", input="x", index=(Var("t"), Literal(0))), _fill(_copy("_desc_x", "_x_smem"), _copy("_desc_w", "_w_smem"))
            ),
            tensors=_TENSORS,
            constant_buffers=frozenset({"w"}),
        )
    )
    needles = ("griddepcontrol.launch_dependents", "_desc_w,", "griddepcontrol.wait", "x[", "_desc_x,")
    release, w_copy, wait, x_read, x_copy = (src.index(n) for n in needles)
    assert release < w_copy < wait < x_read < x_copy
    # The arm stays with the early half and counts both halves' bytes.
    assert src.count("mbarrier_arrive_expect_tx(") == 1
    assert src.index("mbarrier_arrive_expect_tx") < w_copy


def test_without_constants_the_wait_follows_the_setup_and_nothing_moves() -> None:
    src = _body(render_kernelop(_kernel(_fill(_copy("_desc_w", "_w_smem"))), tensors=_TENSORS))
    assert src.index("griddepcontrol.wait") < src.index("_desc_w,")
    assert src.index("mbarrier_init") < src.index("griddepcontrol.wait")


def test_a_fill_stays_behind_a_statement_that_names_its_slab() -> None:
    touch = Cond(
        cond=_TID0,
        body=(
            Load(name="v", input="x", index=(Var("t"), Literal(0))),
            Write(output="_w_smem", index=(Literal(0), Literal(0)), values=("v",)),
        ),
    )
    src = _body(render_kernelop(_kernel(touch, _fill(_copy("_desc_w", "_w_smem"))), tensors=_TENSORS, constant_buffers=frozenset({"w"})))
    assert src.index("griddepcontrol.wait") < src.index("x[") < src.index("_desc_w,")


def test_a_cp_async_group_of_constants_moves_whole_and_a_mixed_one_stays() -> None:
    def copy(src: str) -> CpAsyncCopy:
        return CpAsyncCopy(smem="_w_smem", smem_index=(Literal(0), Var("t")), src=src, src_index=(Literal(0), Var("t")), nbytes=4)

    stat = Load(name="v", input="x", index=(Var("t"), Literal(0)))
    moved = _body(
        render_kernelop(_kernel(stat, copy("w"), copy("w"), CpAsyncCommit()), tensors=_TENSORS, constant_buffers=frozenset({"w"}))
    )
    assert moved.index("&w[") < moved.index("griddepcontrol.wait") < moved.index("x[")
    mixed = _body(
        render_kernelop(_kernel(stat, copy("w"), copy("x"), CpAsyncCommit()), tensors=_TENSORS, constant_buffers=frozenset({"w"}))
    )
    assert mixed.index("griddepcontrol.wait") < mixed.index("&w[")


def test_a_body_that_is_not_one_tile_waits_first() -> None:
    loop = StridedLoop(
        axis=Axis("i", 4), start=Literal(0), step=Literal(1), body=Body((Load(name="v", input="w", index=(Var("i"), Literal(0))),))
    )
    src = _body(render_kernelop(KernelOp(body=Body((loop,)), name="k_plain"), tensors=_TENSORS, constant_buffers=frozenset({"w"})))
    assert src.index("griddepcontrol.launch_dependents") < src.index("griddepcontrol.wait") < src.index("w[")


def test_the_release_is_the_launch_argument_after_the_runtime_ints() -> None:
    """The runtime passes 1 only when the program's next launch may start early: past the last launch the host's
    next grid may launch programmatically without expecting an early release."""
    src = render_kernelop(_kernel(), tensors=_TENSORS, runtime_args=("num_tokens",))
    assert "int num_tokens, int _pdl_release)" in src
    assert 'if (_pdl_release) asm volatile("griddepcontrol.launch_dependents;");' in src


def test_only_constants_are_read_through_the_non_coherent_path() -> None:
    """A kernel released early starts while the grid ahead still writes its other inputs, and the wait makes those
    writes visible to coherent loads only. ``__restrict__`` on a read-only input lets the compiler use the
    non-coherent read-only path (``LDG.CONSTANT``), which served a stale value, so only constants keep it."""
    src = render_kernelop(_kernel(Load(name="v", input="x", index=(Var("t"), Literal(0)))), tensors=_TENSORS, constant_buffers=frozenset({"w"}))
    assert "const __half* __restrict__ w," in src
    assert "const __half* x," in src
    assert "float* __restrict__ y" in src
