SIM ?= icarus
TOPLEVEL_LANG ?= verilog

RTL_DIR = $(PWD)/../../rtl

VERILOG_SOURCES = \
    $(RTL_DIR)/src/top.sv \
    $(RTL_DIR)/Control/Control.sv \
    $(RTL_DIR)/ArgMaxSync/ArgMaxSync.sv \
    $(RTL_DIR)/SortingUnit/SortingUnit.sv

COMPILE_ARGS += -g2012

TOPLEVEL = top
MODULE   = test_top

include $(shell cocotb-config --makefiles)/Makefile.sim
