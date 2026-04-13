# Cadence Genus script - ArgMax / Top
# Usage:
#   genus -f genus_synth.tcl
#
# Requires: LIB_PATH environment variable pointing to the .lib file
# (e.g., sky130_fd_sc_hd__tt_025C_1v80.lib or a PDK std-cell lib).

set TOP     [getenv TARGET]      ;# "ArgMax" or "ReformerTop"
set LIB     [getenv LIB_PATH]

set_db library $LIB
set_db init_hdl_search_path {../rtl/ArgMax ../rtl/SortingUnit ../verification}

if {$TOP == "ArgMax"} {
    read_hdl -sv {ArgMax.sv}
} else {
    read_hdl -sv {ArgMax.sv SortingUnit.sv ReformerTop_reference.sv}
}

elaborate $TOP
current_design $TOP

# Constraints
create_clock -name clk -period 25.0 [get_ports clk]
set_false_path -from [get_ports rst_n]
set_input_delay  -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]

syn_generic
syn_map
syn_opt

file mkdir reports
report_area            > reports/${TOP}_area.rpt
report_timing          > reports/${TOP}_timing.rpt
report_power           > reports/${TOP}_power.rpt
report_gates           > reports/${TOP}_gates.rpt

write_hdl               > reports/${TOP}_netlist.v
write_sdc               > reports/${TOP}.sdc

exit
