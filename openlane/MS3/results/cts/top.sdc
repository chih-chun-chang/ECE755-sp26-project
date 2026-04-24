###############################################################################
# Created by write_sdc
# Fri Apr 24 21:53:16 2026
###############################################################################
current_design top
###############################################################################
# Timing Constraints
###############################################################################
create_clock -name clk -period 25.0000 [get_ports {clk}]
set_clock_transition 0.1500 [get_clocks {clk}]
set_clock_uncertainty 0.2500 clk
set_propagated_clock [get_clocks {clk}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_valid}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value0[0]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value0[1]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value0[2]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value1[0]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value1[1]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value1[2]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value2[0]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value2[1]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value2[2]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value3[0]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value3[1]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value3[2]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value4[0]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value4[1]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {i_value4[2]}]
set_input_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {rst_n}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_bucket[0]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_bucket[1]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_bucket[2]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_index[0]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_index[1]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_index[2]}]
set_output_delay 5.0000 -clock [get_clocks {clk}] -add_delay [get_ports {o_valid}]
###############################################################################
# Environment
###############################################################################
set_load -pin_load 0.0334 [get_ports {o_valid}]
set_load -pin_load 0.0334 [get_ports {o_bucket[2]}]
set_load -pin_load 0.0334 [get_ports {o_bucket[1]}]
set_load -pin_load 0.0334 [get_ports {o_bucket[0]}]
set_load -pin_load 0.0334 [get_ports {o_index[2]}]
set_load -pin_load 0.0334 [get_ports {o_index[1]}]
set_load -pin_load 0.0334 [get_ports {o_index[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {clk}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_valid}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {rst_n}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value0[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value0[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value0[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value1[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value1[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value1[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value2[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value2[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value2[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value3[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value3[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value3[0]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value4[2]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value4[1]}]
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin {Y} -input_transition_rise 0.0000 -input_transition_fall 0.0000 [get_ports {i_value4[0]}]
set_timing_derate -early 0.9500
set_timing_derate -late 1.0500
###############################################################################
# Design Rules
###############################################################################
set_max_transition 0.7500 [current_design]
set_max_fanout 128.0000 [current_design]
