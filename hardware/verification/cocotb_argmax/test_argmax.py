"""Cocotb testbench for ArgMax unit. Compares against Python golden model.

Run with:
  pip install cocotb cocotb-test
  cd hardware/verification/cocotb_argmax && make
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


async def reset(dut):
    dut.rst_n.value = 0
    dut.i_index.value = 0
    dut.i_value.value = 0
    await Timer(20, units="ns")
    dut.rst_n.value = 1
    await Timer(10, units="ns")


def golden(vec):
    mx, bi = 0, 0
    for i, v in enumerate(vec):
        if v > mx:
            mx, bi = v, i
    return mx, bi


@cocotb.test()
async def test_basic(dut):
    cocotb.start_soon(Clock(dut.i_d_clk, 10, units="ns").start())
    await reset(dut)
    cases = [
        [1,2,3,4,5,6,7,0],
        [7,6,5,4,3,2,1,0],
        [4,4,4,4,4,4,4,4],
        [0,0,0,0,0,0,0,6],
    ]
    for vec in cases:
        await reset(dut)
        for i, v in enumerate(vec):
            await FallingEdge(dut.i_d_clk)
            dut.i_index.value = i
            dut.i_value.value = v
        await RisingEdge(dut.i_d_clk)
        await Timer(1, units="ns")
        exp_mx, exp_bi = golden(vec)
        assert int(dut.o_max.value) == exp_mx, f"max mismatch {dut.o_max.value} vs {exp_mx}"
        assert int(dut.o_b_i.value) == exp_bi, f"b_i mismatch {dut.o_b_i.value} vs {exp_bi}"


@cocotb.test()
async def test_random(dut):
    cocotb.start_soon(Clock(dut.i_d_clk, 10, units="ns").start())
    random.seed(0xC001)
    for _ in range(50):
        vec = [random.randint(0, 7) for _ in range(8)]
        await reset(dut)
        for i, v in enumerate(vec):
            await FallingEdge(dut.i_d_clk)
            dut.i_index.value = i
            dut.i_value.value = v
        await RisingEdge(dut.i_d_clk)
        await Timer(1, units="ns")
        exp_mx, exp_bi = golden(vec)
        assert int(dut.o_max.value) == exp_mx
        assert int(dut.o_b_i.value) == exp_bi
