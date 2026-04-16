`default_nettype none

module SortingUnit_tb();

    logic       clk;
    logic       rst_n;
    logic       i_start_sort;
    logic [2:0] i_b_i0;
    logic [2:0] i_b_i1;
    logic [2:0] i_b_i2;
    logic [2:0] i_b_i3;
    logic [2:0] i_b_i4;
    wire  [2:0] o_index;
    wire  [2:0] o_bucket;
    wire        o_valid;

    SortingUnit dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_start_sort(i_start_sort),
        .i_b_i0      (i_b_i0),
        .i_b_i1      (i_b_i1),
        .i_b_i2      (i_b_i2),
        .i_b_i3      (i_b_i3),
        .i_b_i4      (i_b_i4),
        .o_index     (o_index),
        .o_bucket    (o_bucket),
        .o_valid     (o_valid)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task do_reset();
        begin
            rst_n        = 1'b0;
            i_start_sort = 1'b0;
            i_b_i0       = 3'b0;
            i_b_i1       = 3'b0;
            i_b_i2       = 3'b0;
            i_b_i3       = 3'b0;
            i_b_i4       = 3'b0;
            repeat (2) @(posedge clk);
            rst_n = 1'b1;
            @(posedge clk);
        end
    endtask

    task load_batch(
        input logic [2:0] b0,
        input logic [2:0] b1,
        input logic [2:0] b2,
        input logic [2:0] b3,
        input logic [2:0] b4
    );
        begin
            @(negedge clk);
            i_b_i0       = b0;
            i_b_i1       = b1;
            i_b_i2       = b2;
            i_b_i3       = b3;
            i_b_i4       = b4;
            i_start_sort = 1'b1;
            @(negedge clk);
            i_start_sort = 1'b0;
        end
    endtask

    task expect_output(
        input logic [2:0] exp_index,
        input logic [2:0] exp_bucket,
        input string label
    );
        begin
            forever begin
                @(posedge clk);
                if (o_valid) begin
                    if (o_index === exp_index && o_bucket === exp_bucket)
                        $display("PASS [%s] index=%0d bucket=%0d", label, o_index, o_bucket);
                    else
                        $display("FAIL [%s] expected index=%0d bucket=%0d got index=%0d bucket=%0d",
                                 label, exp_index, exp_bucket, o_index, o_bucket);
                    disable expect_output;
                end
            end
        end
    endtask

    initial begin
        $display("===== SortingUnit testbench =====");

        // Test 1: mixed buckets, including ties.
        // Inputs: idx0->3, idx1->1, idx2->3, idx3->0, idx4->1
        // Expected output order: (3,0), (1,1), (4,1), (0,3), (2,3)
        do_reset();
        load_batch(3'd3, 3'd1, 3'd3, 3'd0, 3'd1);
        expect_output(3'd3, 3'd0, "mixed #1");
        expect_output(3'd1, 3'd1, "mixed #2");
        expect_output(3'd4, 3'd1, "mixed #3");
        expect_output(3'd0, 3'd3, "mixed #4");
        expect_output(3'd2, 3'd3, "mixed #5");

        // Test 2: all in same bucket, expect ascending indices.
        do_reset();
        load_batch(3'd2, 3'd2, 3'd2, 3'd2, 3'd2);
        expect_output(3'd0, 3'd2, "same bucket #1");
        expect_output(3'd1, 3'd2, "same bucket #2");
        expect_output(3'd2, 3'd2, "same bucket #3");
        expect_output(3'd3, 3'd2, "same bucket #4");
        expect_output(3'd4, 3'd2, "same bucket #5");

        // Test 3: already sorted distinct buckets.
        do_reset();
        load_batch(3'd0, 3'd1, 3'd2, 3'd3, 3'd4);
        expect_output(3'd0, 3'd0, "distinct #1");
        expect_output(3'd1, 3'd1, "distinct #2");
        expect_output(3'd2, 3'd2, "distinct #3");
        expect_output(3'd3, 3'd3, "distinct #4");
        expect_output(3'd4, 3'd4, "distinct #5");

        repeat (5) @(posedge clk);
        $finish;
    end

endmodule
