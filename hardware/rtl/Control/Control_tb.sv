`default_nettype none

module Control_tb();

    // ----------------------------------------------------------------
    // Parameters (must match DUT defaults)
    // ----------------------------------------------------------------
    localparam int B     = 8;
    localparam int IDX_W = 3;

    // ----------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------
    logic             clk;
    logic             rst_n;
    logic             i_valid;
    wire  [IDX_W-1:0] o_index;
    wire              o_start_sort;

    // ----------------------------------------------------------------
    // DUT instantiation
    // ----------------------------------------------------------------
    Control #(
        .B     (B),
        .IDX_W (IDX_W)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_valid     (i_valid),
        .o_index     (o_index),
        .o_start_sort(o_start_sort)
    );

    // ----------------------------------------------------------------
    // Clock: 10 ns period
    // ----------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // ----------------------------------------------------------------
    // Pass/fail counters
    // ----------------------------------------------------------------
    int pass_count = 0;
    int fail_count = 0;

    // ----------------------------------------------------------------
    // Task: assert/release reset
    // ----------------------------------------------------------------
    task do_reset();
        rst_n   = 0;
        i_valid = 0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1;
        @(negedge clk);   // settle
    endtask

    // ----------------------------------------------------------------
    // Task: drive i_valid for one cycle (change on negedge)
    // ----------------------------------------------------------------
    task drive_valid(input logic val);
        @(negedge clk);
        i_valid = val;
    endtask

    // ----------------------------------------------------------------
    // Task: generic checker, samples 1 ns after posedge
    // ----------------------------------------------------------------
    task check_outputs(
        input logic [IDX_W-1:0] exp_index,
        input logic              exp_sort,
        input string             label
    );
        @(posedge clk); #1;
        if (o_index === exp_index && o_start_sort === exp_sort) begin
            $display("PASS  [%s]  o_index=%0d  o_start_sort=%0b",
                     label, o_index, o_start_sort);
            pass_count++;
        end else begin
            $display("FAIL  [%s]  expected index=%0d sort=%0b  got index=%0d sort=%0b",
                     label, exp_index, exp_sort, o_index, o_start_sort);
            fail_count++;
        end
    endtask

    // ----------------------------------------------------------------
    // Stimulus
    // ----------------------------------------------------------------
    initial begin
        $display("========== Control Testbench (B=%0d, IDX_W=%0d) ==========", B, IDX_W);

        // ---- Test 1: Reset clears outputs ----------------------------
        // After rst_n is deasserted, o_index should be 0 and
        // o_start_sort should be 0.
        do_reset();
        @(posedge clk); #1;
        if (o_index === '0 && o_start_sort === 1'b0) begin
            $display("PASS  [reset]  o_index=0  o_start_sort=0");
            pass_count++;
        end else begin
            $display("FAIL  [reset]  expected 0/0  got index=%0d sort=%0b",
                     o_index, o_start_sort);
            fail_count++;
        end

        // ---- Test 2: o_index increments only on i_valid --------------
        // Drive 4 valid cycles and confirm o_index advances each time.
        do_reset();
        begin
            int ok;
            ok = 1;
            for (int i = 0; i < 4; i++) begin
                drive_valid(1'b1);       // drive at negedge
                @(posedge clk); #1;      // sample after posedge
                // After the (i+1)-th valid posedge, idx_cnt == i+1
                if (o_index !== IDX_W'(i + 1)) begin
                    $display("FAIL  [index increment, step %0d]  expected %0d  got %0d",
                             i, i+1, o_index);
                    fail_count++;
                    ok = 0;
                end
            end
            if (ok) begin
                $display("PASS  [index increment]  o_index advanced correctly 1..4");
                pass_count++;
            end
        end

        // ---- Test 3: No increment when i_valid deasserted ------------
        // Hold valid low for 3 cycles; index must stay frozen.
        do_reset();
        drive_valid(1'b1);               // index becomes 1
        @(posedge clk); #1;
        begin
            logic [IDX_W-1:0] frozen;
            frozen = o_index;            // should be 1
            for (int i = 0; i < 3; i++) begin
                drive_valid(1'b0);
                @(posedge clk); #1;
                if (o_index !== frozen) begin
                    $display("FAIL  [no increment on idle, cycle %0d]  expected %0d got %0d",
                             i, frozen, o_index);
                    fail_count++;
                    frozen = IDX_W'('x); // poison so future cycles also fail
                end
            end
            if (frozen === IDX_W'(1)) begin
                $display("PASS  [no increment on idle]  o_index held at 1 for 3 idle cycles");
                pass_count++;
            end
        end

        // ---- Test 4: Full batch, start_sort pulses once --------------
        // Stream exactly B=8 valid cycles.  start_sort must be 0 for
        // cycles 0..6, then 1 exactly one cycle after the 8th valid,
        // then return to 0.
        do_reset();
        begin
            // Cycles 0..6: valid, no sort pulse yet
            for (int i = 0; i < B-1; i++) begin
                drive_valid(1'b1);
                @(posedge clk); #1;
                if (o_start_sort !== 1'b0) begin
                    $display("FAIL  [sort not early, cycle %0d]  o_start_sort=%0b", i, o_start_sort);
                    fail_count++;
                end
            end
            // Cycle 7 (final valid, idx_cnt==B-1 at start of cycle)
            drive_valid(1'b1);
            @(posedge clk); #1;     // now start_sort should be 1
            if (o_start_sort === 1'b1) begin
                $display("PASS  [start_sort pulse]  o_start_sort=1 after final valid");
                pass_count++;
            end else begin
                $display("FAIL  [start_sort pulse]  expected 1, got %0b", o_start_sort);
                fail_count++;
            end
            // One more cycle with valid deasserted — pulse must drop
            drive_valid(1'b0);
            @(posedge clk); #1;
            if (o_start_sort === 1'b0) begin
                $display("PASS  [start_sort one-cycle]  o_start_sort dropped back to 0");
                pass_count++;
            end else begin
                $display("FAIL  [start_sort one-cycle]  o_start_sort still %0b", o_start_sort);
                fail_count++;
            end
        end

        // ---- Test 5: start_sort requires i_valid on last cycle -------
        // Advance to idx_cnt==B-1 but deassert i_valid on that cycle.
        // start_sort must NOT fire.
        do_reset();
        for (int i = 0; i < B-1; i++) begin   // arrive at idx_cnt == B-1
            drive_valid(1'b1);
            @(posedge clk);
        end
        drive_valid(1'b0);                     // hold valid low at idx_cnt == B-1
        @(posedge clk); #1;
        if (o_start_sort === 1'b0) begin
            $display("PASS  [no spurious sort]  start_sort suppressed when valid low at final index");
            pass_count++;
        end else begin
            $display("FAIL  [no spurious sort]  start_sort fired without valid at final index");
            fail_count++;
        end

        // ---- Test 6: Valid with gaps, still reaches B ----------------
        // Interleave valid/idle so total valid cycles == B.
        // start_sort should fire the same way as a contiguous burst.
        do_reset();
        begin
            int valids_driven;
            valids_driven = 0;
            for (int i = 0; i < B*2; i++) begin
                // drive valid on even loop iterations
                drive_valid(i[0] == 0 ? 1'b1 : 1'b0);
                @(posedge clk); #1;
                if (i[0] == 0) valids_driven++;
                if (valids_driven == B) begin
                    if (o_start_sort === 1'b1) begin
                        $display("PASS  [valid with gaps]  start_sort fired after %0d valid cycles", B);
                        pass_count++;
                    end else begin
                        $display("FAIL  [valid with gaps]  start_sort=%0b after %0d valid cycles",
                                 o_start_sort, B);
                        fail_count++;
                    end
                    break;
                end
            end
        end

        // ---- Test 7: Mid-batch reset clears state --------------------
        // Push a few valid cycles, assert reset, release; index must
        // return to 0 and start_sort must be 0.
        do_reset();
        drive_valid(1'b1);
        drive_valid(1'b1);
        drive_valid(1'b1);          // idx_cnt should be 3 here
        @(posedge clk);
        do_reset();                 // mid-stream reset
        @(posedge clk); #1;
        if (o_index === '0 && o_start_sort === 1'b0) begin
            $display("PASS  [mid-batch reset]  o_index=0  o_start_sort=0 after reset");
            pass_count++;
        end else begin
            $display("FAIL  [mid-batch reset]  expected 0/0  got index=%0d sort=%0b",
                     o_index, o_start_sort);
            fail_count++;
        end

        // ---- Summary -------------------------------------------------
        @(negedge clk);
        $display("======================================================");
        $display("Results: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("======================================================");
        $finish;
    end

endmodule
