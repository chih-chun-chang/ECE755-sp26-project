/*
    Top-level for the Reformer LSH hardware accelerator.

    Architecture:
      - Control block drives the shared index counter and emits start_sort
        one cycle after the final i_valid of a batch.
      - 5 parallel ArgMaxSync units each track the maximum of one projected
        dimension across B=8 hash keys; they output the winning bucket ID.
      - SortingUnit latches the 5 bucket IDs on start_sort and streams
        sorted (index, bucket) pairs on o_index / o_bucket / o_valid.

    Parameters:
      B     - number of hash buckets / input vectors per batch  (default 8)
      IDX_W - bit-width of the index / bucket fields            (default 3)
      VAL_W - bit-width of each projected value                 (default 3)
*/

`default_nettype none

module top #(
    parameter int B     = 8,
    parameter int IDX_W = 3,
    parameter int VAL_W = 3
)(
    input  wire             clk,
    input  wire             rst_n,

    // Host interface: assert i_valid with the next 5 projected values
    input  wire             i_valid,
    input  wire [VAL_W-1:0] i_value0,   // projected value for ArgMax unit 0
    input  wire [VAL_W-1:0] i_value1,   // projected value for ArgMax unit 1
    input  wire [VAL_W-1:0] i_value2,   // projected value for ArgMax unit 2
    input  wire [VAL_W-1:0] i_value3,   // projected value for ArgMax unit 3
    input  wire [VAL_W-1:0] i_value4,   // projected value for ArgMax unit 4

    // Sorted output stream
    output wire [IDX_W-1:0] o_index,    // which ArgMax unit (0-4)
    output wire [IDX_W-1:0] o_bucket,   // bucket ID assigned to that unit
    output wire             o_valid     // o_index / o_bucket are valid this cycle
);

    // -----------------------------------------------------------------------
    // Internal wires
    // -----------------------------------------------------------------------
    wire [IDX_W-1:0] ctrl_index;        // shared hash-key index from Control
    wire             ctrl_start_sort;   // one-cycle pulse: begin sort phase

    wire [IDX_W-1:0] b_i [0:4];         // winning bucket ID from each ArgMaxSync

    // -----------------------------------------------------------------------
    // Control block
    // -----------------------------------------------------------------------
    Control #(
        .B     (B),
        .IDX_W (IDX_W)
    ) u_control (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_valid     (i_valid),
        .o_index     (ctrl_index),
        .o_start_sort(ctrl_start_sort)
    );

    // -----------------------------------------------------------------------
    // 5 parallel ArgMaxSync units
    // -----------------------------------------------------------------------
    ArgMaxSync u_am0 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (ctrl_index),
        .i_value (i_value0),
        .o_max   (/* unused */),
        .o_b_i   (b_i[0])
    );

    ArgMaxSync u_am1 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (ctrl_index),
        .i_value (i_value1),
        .o_max   (/* unused */),
        .o_b_i   (b_i[1])
    );

    ArgMaxSync u_am2 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (ctrl_index),
        .i_value (i_value2),
        .o_max   (/* unused */),
        .o_b_i   (b_i[2])
    );

    ArgMaxSync u_am3 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (ctrl_index),
        .i_value (i_value3),
        .o_max   (/* unused */),
        .o_b_i   (b_i[3])
    );

    ArgMaxSync u_am4 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (ctrl_index),
        .i_value (i_value4),
        .o_max   (/* unused */),
        .o_b_i   (b_i[4])
    );

    // -----------------------------------------------------------------------
    // Sorting Unit
    // -----------------------------------------------------------------------
    SortingUnit u_sort (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_start_sort(ctrl_start_sort),
        .i_b_i0      (b_i[0]),
        .i_b_i1      (b_i[1]),
        .i_b_i2      (b_i[2]),
        .i_b_i3      (b_i[3]),
        .i_b_i4      (b_i[4]),
        .o_index     (o_index),
        .o_bucket    (o_bucket),
        .o_valid     (o_valid)
    );

endmodule
