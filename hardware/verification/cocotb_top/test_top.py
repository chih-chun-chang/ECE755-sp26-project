"""Cocotb full-pipeline testbench for the Reformer LSH accelerator.

Drives random 5x8 batches through Control + 5x ArgMaxSync + SortingUnit and
compares the emitted (index, bucket) stream against software/emulation/
golden_model.py (the bit-accurate Python reference).

Run with:
  pip install cocotb
  cd hardware/verification/cocotb_top && make
"""
import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Import the Python golden model directly (source of truth).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "software", "emulation"))
import golden_model as gm  # noqa: E402

CLK_PERIOD_NS = 10


async def reset(dut):
    dut.rst_n.value = 0
    dut.i_valid.value = 0
    for k in range(5):
        getattr(dut, f"i_value{k}").value = 0
    await Timer(2 * CLK_PERIOD_NS, units="ns")
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_batch(dut, vectors):
    """vectors: list[N_VEC] of list[B] of ints. Drives one column per clk."""
    for t in range(gm.B):
        for u in range(gm.N_VEC):
            getattr(dut, f"i_value{u}").value = vectors[u][t]
        dut.i_valid.value = 1
        await RisingEdge(dut.clk)
    dut.i_valid.value = 0
    for u in range(gm.N_VEC):
        getattr(dut, f"i_value{u}").value = 0


async def collect_outputs(dut, max_cycles):
    """Sample o_valid/o_index/o_bucket on each posedge for max_cycles."""
    out = []
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")  # settle combinational o_valid
        if int(dut.o_valid.value) == 1:
            out.append((int(dut.o_index.value), int(dut.o_bucket.value)))
    return out


async def run_one(dut, vectors, label=""):
    expected = gm.top_block(vectors)
    await reset(dut)
    await drive_batch(dut, vectors)
    got = await collect_outputs(dut, 2 * (gm.N_VEC + gm.B))
    assert got == expected, (
        f"[{label}] mismatch\n  bids={gm.argmax_stage(vectors)}\n"
        f"  got={got}\n  exp={expected}\n  vectors={vectors}"
    )


@cocotb.test()
async def test_directed(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    cases = {
        "all_zero":        [[0] * gm.B for _ in range(gm.N_VEC)],
        "ramps":           [[t for t in range(gm.B)] for _ in range(gm.N_VEC)],
        "same_bucket":     [[0, 0, 0, 0, 0, 0, 0, 3] for _ in range(gm.N_VEC)],
        "distinct":        [
            [1, 2, 3, 4, 5, 6, 7, 0],   # -> bucket 6
            [7, 6, 5, 4, 3, 2, 1, 0],   # -> bucket 0
            [2, 5, 1, 3, 0, 0, 0, 0],   # -> bucket 1
            [4, 4, 4, 4, 4, 4, 4, 4],   # -> bucket 0
            [0, 0, 0, 0, 0, 0, 0, 6],   # -> bucket 7
        ],
    }
    for label, vecs in cases.items():
        await run_one(dut, vecs, label)


@cocotb.test()
async def test_random(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    random.seed(0xC001_D00D)
    N_TRIAL = 200
    for t in range(N_TRIAL):
        vectors = [
            [random.randint(0, gm.VMAX) for _ in range(gm.B)]
            for _ in range(gm.N_VEC)
        ]
        await run_one(dut, vectors, label=f"rand{t}")
    dut._log.info(f"Random trials passed: {N_TRIAL}")
