/*
    ArgMax Unit for reformer hardware accelerator
    Inputs:
        -i_index: From Control Block, shared between multiple units
        -i_value: From external microcontroller, only real input
        -i_d_clk: From external microcontroller, external clock, high when next set of values are ready
        -rst_n: global reset
    Outputs:
        -o_max: Used for testing, output not actually used
        -o_b_i: index of largest value, for bucket sorting
*/

`default_nettype none

module ArgMax(
    input wire [2:0] i_index,
    input wire [2:0] i_value,
    input wire i_d_clk,
    input wire rst_n,
    output wire [2:0] o_max,
    output wire [2:0] o_b_i
);

reg [2:0] max, b_i;

always_ff @(posedge i_d_clk or negedge rst_n) begin
    if (!rst_n) begin
        max <= 3'b0;
        b_i <= 3'b0;
    end
    else begin
        if (i_value > max) begin
            max <= i_value;
            b_i <= i_index;
        end
        else begin
            max <= max;
            b_i <= b_i;
        end
    end
end

assign o_max = max;
assign o_b_i = b_i;

endmodule
