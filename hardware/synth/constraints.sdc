# constraints.sdc - timing constraints
# 40 MHz target (25 ns period). Adjust to match achieved Fmax after STA.
create_clock -name clk     -period 25.0 [get_ports clk]
create_clock -name i_d_clk -period 25.0 [get_ports i_d_clk]

# async reset
set_false_path -from [get_ports rst_n]

# conservative I/O timing
set_input_delay  -clock clk -max 2.0 [all_inputs]
set_output_delay -clock clk -max 2.0 [all_outputs]

set_load 0.02 [all_outputs]
