/*
    Control block for reformer LSH accelerator.

    Responsibilities:
      - Drive the shared ArgMax index (idx_cnt) while the host streams values
        under i_valid.
      - Emit a one-cycle start_sort pulse the cycle after the final value
        (idx_cnt == B-1) is captured, so SortingUnit latches the now-stable
        b_i[] outputs and begins streaming sorted results.

    Assumptions:
      - Single batch per rst_n deassertion; host drives exactly B asserted
        i_valid cycles, then stops.
      - Single clock domain (clk); i_valid is synchronous to clk.
*/

`default_nettype none

module Control #(
    parameter int B     = 8,
    parameter int IDX_W = 3
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             i_valid,
    output wire [IDX_W-1:0] o_index,
    output wire             o_start_sort
);

reg [IDX_W-1:0] idx_cnt;
reg             start_sort;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        idx_cnt    <= '0;
        start_sort <= 1'b0;
    end
    else begin
        start_sort <= i_valid && (idx_cnt == IDX_W'(B-1));
        if (i_valid)
            idx_cnt <= idx_cnt + 1'b1;
    end
end

assign o_index      = idx_cnt;
assign o_start_sort = start_sort;

endmodule
