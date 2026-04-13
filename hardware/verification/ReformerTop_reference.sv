// =============================================================
// ReformerTop_reference.sv
// REFERENCE-ONLY top wrapper for the standalone compute block
// (5 parallel ArgMax units + counting-sort Sorting Unit).
//
// NOTE: This file lives under hardware/verification/ and is NOT
// part of hardware/rtl/. It is provided so the team can integrate
// it by copying/moving it into hardware/rtl/Top/ when ready.
//
// This wrapper matches the DR1 pin-out spec:
//   ui_in [7:0] + uio[7:0]  -> 5 x 3-bit data + 1 shared d_clk
//   uo_out[7:0] -> {unused[7], valid, bucket[2:0], index[2:0]}
// =============================================================
`default_nettype none

module ReformerTop (
    input  wire        clk,        // system clock (sort phase)
    input  wire        rst_n,      // global async reset, active low
    input  wire [7:0]  ui_in,      // ArgMax[0..1] data + ArgMax[2] partial + d_clk
    input  wire [7:0]  uio_in,     // ArgMax[2..4] data
    output wire [7:0]  uo_out      // sorted output stream + valid
);

    // -------------------------------------------------------------
    // Pin mapping (example). Adjust to match your final tapeout plan.
    //   ui_in [2:0]  = ArgMax0 value
    //   ui_in [5:3]  = ArgMax1 value
    //   ui_in [6]    = d_clk
    //   ui_in [7]    = start / load_en (batch start pulse)
    //   uio_in[2:0]  = ArgMax2 value
    //   uio_in[5:3]  = ArgMax3 value
    //   uio_in[7:6]  = ArgMax4 value[1:0]   -> NOTE: ArgMax4 MSB shared below
    //   (Simplify for now: assume all 15 data bits + 1 d_clk fit in 16 pins)
    // -------------------------------------------------------------
    wire        d_clk = ui_in[6];
    wire        start = ui_in[7];
    wire [2:0]  v0    = ui_in[2:0];
    wire [2:0]  v1    = ui_in[5:3];
    wire [2:0]  v2    = uio_in[2:0];
    wire [2:0]  v3    = uio_in[5:3];
    wire [2:0]  v4    = {uio_in[6], uio_in[7], 1'b0}; // placeholder; remap as needed

    // -------------------------------------------------------------
    // Shared index counter (driven by d_clk)
    // -------------------------------------------------------------
    reg [2:0] idx_cnt;
    always @(posedge d_clk or negedge rst_n) begin
        if (!rst_n)       idx_cnt <= 3'd0;
        else if (start)   idx_cnt <= 3'd0;
        else              idx_cnt <= idx_cnt + 3'd1;
    end

    // -------------------------------------------------------------
    // 5 parallel ArgMax units
    // -------------------------------------------------------------
    wire [2:0] b_id [0:4];
    wire [2:0] mx   [0:4];

    ArgMax u_am0 (.i_index(idx_cnt), .i_value(v0), .i_d_clk(d_clk), .rst_n(rst_n), .o_max(mx[0]), .o_b_i(b_id[0]));
    ArgMax u_am1 (.i_index(idx_cnt), .i_value(v1), .i_d_clk(d_clk), .rst_n(rst_n), .o_max(mx[1]), .o_b_i(b_id[1]));
    ArgMax u_am2 (.i_index(idx_cnt), .i_value(v2), .i_d_clk(d_clk), .rst_n(rst_n), .o_max(mx[2]), .o_b_i(b_id[2]));
    ArgMax u_am3 (.i_index(idx_cnt), .i_value(v3), .i_d_clk(d_clk), .rst_n(rst_n), .o_max(mx[3]), .o_b_i(b_id[3]));
    ArgMax u_am4 (.i_index(idx_cnt), .i_value(v4), .i_d_clk(d_clk), .rst_n(rst_n), .o_max(mx[4]), .o_b_i(b_id[4]));

    // -------------------------------------------------------------
    // Sorting Unit (behavioral model for verification until RTL
    //              SortingUnit.sv is implemented).
    // This is a BEHAVIORAL placeholder used only for verification.
    // -------------------------------------------------------------
    localparam N = 5;
    localparam B = 8;

    reg [2:0] buf_b [0:N-1];
    reg       valid_flag [0:N-1];
    reg [2:0] bucket_cnt;
    reg       loaded;
    reg [2:0] o_index, o_bucket;
    reg       o_valid;
    integer   i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N; i = i + 1) begin
                buf_b[i]      <= 3'd0;
                valid_flag[i] <= 1'b0;
            end
            bucket_cnt <= 3'd0;
            loaded     <= 1'b0;
            o_valid    <= 1'b0;
            o_index    <= 3'd0;
            o_bucket   <= 3'd0;
        end else begin
            // latch buckets on idx_cnt wrap-around
            if (idx_cnt == 3'd7 && !loaded) begin
                for (i = 0; i < N; i = i + 1) begin
                    buf_b[i]      <= b_id[i];
                    valid_flag[i] <= 1'b1;
                end
                bucket_cnt <= 3'd0;
                loaded     <= 1'b1;
            end else if (loaded) begin
                // priority-encode first matching
                o_valid <= 1'b0;
                for (i = N-1; i >= 0; i = i - 1) begin
                    if (valid_flag[i] && (buf_b[i] == bucket_cnt)) begin
                        o_index       <= i[2:0];
                        o_bucket      <= bucket_cnt;
                        o_valid       <= 1'b1;
                        valid_flag[i] <= 1'b0;
                    end
                end
                // if no match this cycle, advance bucket_cnt
                if (!(|((valid_flag[0] && buf_b[0]==bucket_cnt) ||
                        (valid_flag[1] && buf_b[1]==bucket_cnt) ||
                        (valid_flag[2] && buf_b[2]==bucket_cnt) ||
                        (valid_flag[3] && buf_b[3]==bucket_cnt) ||
                        (valid_flag[4] && buf_b[4]==bucket_cnt))))
                    bucket_cnt <= bucket_cnt + 3'd1;
            end
        end
    end

    assign uo_out = {1'b0, o_valid, o_bucket, o_index};

endmodule
