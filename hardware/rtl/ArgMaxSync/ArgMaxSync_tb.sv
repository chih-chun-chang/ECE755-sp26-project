`default_nettype none

module ArgMaxSync_tb();

    // ----------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------
    logic       clk;
    logic       rst_n;
    logic       i_valid;
    logic [2:0] i_index;
    logic [2:0] i_value;
    wire  [2:0] o_max;
    wire  [2:0] o_b_i;

    // ----------------------------------------------------------------
    // DUT instantiation
    // ----------------------------------------------------------------
    ArgMaxSync dut (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_valid (i_valid),
        .i_index (i_index),
        .i_value (i_value),
        .o_max   (o_max),
        .o_b_i   (o_b_i)
    );

    // ----------------------------------------------------------------
    // Clock: 10 ns period
    // ----------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // ----------------------------------------------------------------
    // Task: reset DUT
    // ----------------------------------------------------------------
    task do_reset();
        rst_n   = 0;
        i_valid = 0;
        i_index = 3'b0;
        i_value = 3'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1;
        @(negedge clk);   // settle
    endtask

    // ----------------------------------------------------------------
    // Task: drive one valid (index, value) pair on negedge
    // ----------------------------------------------------------------
    task drive(input logic [2:0] idx, input logic [2:0] val);
        @(negedge clk);
        i_valid = 1'b1;
        i_index = idx;
        i_value = val;
    endtask

    // ----------------------------------------------------------------
    // Task: drive one invalid cycle (i_valid=0) on negedge
    // ----------------------------------------------------------------
    task drive_invalid(input logic [2:0] idx, input logic [2:0] val);
        @(negedge clk);
        i_valid = 1'b0;
        i_index = idx;
        i_value = val;
    endtask

    // ----------------------------------------------------------------
    // Task: check outputs and report pass/fail
    // ----------------------------------------------------------------
    int pass_count = 0;
    int fail_count = 0;

    task check(
        input logic [2:0] exp_max,
        input logic [2:0] exp_b_i,
        input string       label
    );
        @(posedge clk); #1;   // sample shortly after rising edge
        if (o_max === exp_max && o_b_i === exp_b_i) begin
            $display("PASS  [%s]  o_max=%0d  o_b_i=%0d", label, o_max, o_b_i);
            pass_count++;
        end else begin
            $display("FAIL  [%s]  expected max=%0d b_i=%0d  got max=%0d b_i=%0d",
                     label, exp_max, exp_b_i, o_max, o_b_i);
            fail_count++;
        end
    endtask

    // ----------------------------------------------------------------
    // Stimulus
    // ----------------------------------------------------------------
    initial begin
        $display("========== ArgMaxSync Testbench ==========");

        // ---- Test 1: Reset behaviour ---------------------------------
        // After reset both registers must be 0.
        do_reset();
        @(posedge clk); #1;
        if (o_max === 3'b0 && o_b_i === 3'b0) begin
            $display("PASS  [reset]  o_max=0  o_b_i=0");
            pass_count++;
        end else begin
            $display("FAIL  [reset]  expected 0/0  got max=%0d b_i=%0d", o_max, o_b_i);
            fail_count++;
        end

        // ---- Test 2: Ascending stream --------------------------------
        // Feed values 1,2,3,4,5,6,7 at indices 0..6.
        // Max should track up to 7 at index 6.
        do_reset();
        drive(3'd0, 3'd1);
        drive(3'd1, 3'd2);
        drive(3'd2, 3'd3);
        drive(3'd3, 3'd4);
        drive(3'd4, 3'd5);
        drive(3'd5, 3'd6);
        drive(3'd6, 3'd7);
        check(3'd7, 3'd6, "ascending stream");

        // ---- Test 3: Descending stream --------------------------------
        // Feed values 7,6,5,4,3,2,1 at indices 0..6.
        // Max stays at first value (7) at index 0.
        do_reset();
        drive(3'd0, 3'd7);
        drive(3'd1, 3'd6);
        drive(3'd2, 3'd5);
        drive(3'd3, 3'd4);
        drive(3'd4, 3'd3);
        drive(3'd5, 3'd2);
        drive(3'd6, 3'd1);
        check(3'd7, 3'd0, "descending stream");

        // ---- Test 4: Max in the middle --------------------------------
        // Values: 2,5,1,3 → max=5 at index 1.
        do_reset();
        drive(3'd0, 3'd2);
        drive(3'd1, 3'd5);
        drive(3'd2, 3'd1);
        drive(3'd3, 3'd3);
        check(3'd5, 3'd1, "max in middle");

        // ---- Test 5: Equal values — first occurrence wins ------------
        // DUT only updates on strictly greater; index 0 must be retained.
        do_reset();
        drive(3'd0, 3'd4);
        drive(3'd1, 3'd4);
        drive(3'd2, 3'd4);
        check(3'd4, 3'd0, "equal values - first wins");

        // ---- Test 6: Single element -----------------------------------
        do_reset();
        drive(3'd5, 3'd6);
        check(3'd6, 3'd5, "single element");

        // ---- Test 7: Re-reset mid-stream clears state ----------------
        // Push a high value, reset, push a lower value; expect lower.
        do_reset();
        drive(3'd2, 3'd7);
        @(negedge clk);   // let it register
        do_reset();
        drive(3'd3, 3'd3);
        check(3'd3, 3'd3, "re-reset clears state");

        // ---- Test 8: Max at last index --------------------------------
        // Values: 1,2,1,2,1,2,7 — spike at the very end.
        do_reset();
        drive(3'd0, 3'd1);
        drive(3'd1, 3'd2);
        drive(3'd2, 3'd1);
        drive(3'd3, 3'd2);
        drive(3'd4, 3'd1);
        drive(3'd5, 3'd2);
        drive(3'd6, 3'd7);
        check(3'd7, 3'd6, "max at last index");

        // ---- Test 9: i_valid=0 suppresses updates --------------------
        // Drive a large value with valid deasserted; state must not change.
        do_reset();
        drive(3'd0, 3'd3);          // valid: max=3, b_i=0
        @(posedge clk);             // let it register
        drive_invalid(3'd1, 3'd7);  // invalid: should be ignored
        drive_invalid(3'd2, 3'd7);
        check(3'd3, 3'd0, "i_valid suppresses update");

        // ---- Test 10: Valid interleaved with invalid cycles ----------
        // Only valid cycles should advance the running max.
        do_reset();
        drive(3'd0, 3'd2);          // valid:   max=2
        drive_invalid(3'd1, 3'd6);  // invalid: ignored
        drive(3'd2, 3'd5);          // valid:   max=5
        drive_invalid(3'd3, 3'd7);  // invalid: ignored
        check(3'd5, 3'd2, "interleaved valid/invalid");

        // ---- Summary -------------------------------------------------
        @(negedge clk);
        $display("==========================================");
        $display("Results: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("==========================================");
        $finish;
    end

endmodule
