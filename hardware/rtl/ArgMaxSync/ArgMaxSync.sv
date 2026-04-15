/*
    ArgMax Unit for reformer hardware accelerator
    Inputs:
        -clk:     system clock
        -rst_n:   global async reset, active low
        -i_valid: host asserts when a new (i_value, i_index) pair is being presented
        -i_index: shared index from Control block
        -i_value: projected component from host
    Outputs:
        -o_max:   current running maximum (exposed for verification)
        -o_b_i:   index of largest value seen so far (bucket ID)
*/

`default_nettype none

module ArgMaxSync(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_valid,
    input  wire [2:0] i_index,
    input  wire [2:0] i_value,
    output wire [2:0] o_max,
    output wire [2:0] o_b_i
);

reg [2:0] max, b_i;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        max <= 3'b0;
        b_i <= 3'b0;
    end
    else if (i_valid) begin
        if (i_value > max) begin
            max <= i_value;
            b_i <= i_index;
        end
    end
end

assign o_max = max;
assign o_b_i = b_i;

endmodule
