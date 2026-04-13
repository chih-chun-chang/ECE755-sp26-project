# Synopsys Design Compiler script - ArgMax / Top
# Usage:
#   dc_shell -f dc_synth.tcl
#
# Set env vars before running:
#   setenv TECH_LIB /path/to/std_cells.db      (e.g., SAED32 or sky130)
#   setenv TARGET   ArgMax                     (or ReformerTop)

set target_library  [getenv TECH_LIB]
set link_library    "* $target_library"
set search_path     [list . ../rtl/ArgMax ../rtl/SortingUnit ../verification]

set TOP [getenv TARGET]

# -------------------------------------------------
# Read RTL
# -------------------------------------------------
if {$TOP == "ArgMax"} {
    analyze -format sverilog {ArgMax.sv}
} else {
    analyze -format sverilog {ArgMax.sv SortingUnit.sv ReformerTop_reference.sv}
}
elaborate $TOP
current_design $TOP
link

# -------------------------------------------------
# Constraints
# -------------------------------------------------
create_clock -name clk -period 25.0 [get_ports clk]
set_false_path -from [get_ports rst_n]
set_max_fanout 8  [current_design]
set_max_transition 0.5 [current_design]

# -------------------------------------------------
# Compile
# -------------------------------------------------
compile_ultra -no_autoungroup

# -------------------------------------------------
# Reports
# -------------------------------------------------
file mkdir reports
report_area            > reports/${TOP}_area.rpt
report_timing -max_paths 5 > reports/${TOP}_timing.rpt
report_power           > reports/${TOP}_power.rpt
report_qor             > reports/${TOP}_qor.rpt

write -format ddc -hier -output reports/${TOP}.ddc
write -format verilog -hier -output reports/${TOP}_netlist.v
write_sdc -nosplit reports/${TOP}.sdc

exit
