// =============================================================
// ArgMax_array_tb.sv
// Integration testbench: instantiates 5x ArgMax units in parallel
// (the "ArgMax stage" of the standalone compute block) and
// self-checks against a SystemVerilog golden model.
//
// Run with iverilog:
//   iverilog -g2012 -o sim_argmax_array \
//       ../rtl/ArgMax/ArgMax.sv ArgMax_array_tb.sv
//   vvp sim_argmax_array
// =============================================================
`default_nettype none
`timescale 1ns/1ps

module ArgMax_array_tb;

    localparam N_VEC = 5;
    localparam B     = 8;
    localparam W     = 3;

    logic       d_clk;
    logic       rst_n;
    logic [2:0] idx_cnt;
    logic [2:0] val [N_VEC];
    wire  [2:0] bucket [N_VEC];
    wire  [2:0] mx     [N_VEC];

    // generate units
    genvar g;
    generate
        for (g = 0; g < N_VEC; g = g + 1) begin : GEN_AM
            ArgMax u (
                .i_index (idx_cnt),
                .i_value (val[g]),
                .i_d_clk (d_clk),
                .rst_n   (rst_n),
                .o_max   (mx[g]),
                .o_b_i   (bucket[g])
            );
        end
    endgenerate

    // 10 ns clock
    initial d_clk = 0;
    always #5 d_clk = ~d_clk;

    // shared index counter
    always @(posedge d_clk or negedge rst_n)
        if (!rst_n) idx_cnt <= 3'd0;
        else        idx_cnt <= idx_cnt + 3'd1;

    task do_reset;
        begin
            rst_n = 0;
            for (int i = 0; i < N_VEC; i++) val[i] = 3'd0;
            @(negedge d_clk);
            @(negedge d_clk);
            rst_n = 1;
            @(negedge d_clk);
        end
    endtask

    // SV golden model
    function automatic [2:0] golden_argmax(input [W-1:0] v [B]);
        logic [2:0] mx_val; logic [2:0] mx_idx;
        mx_val = 0; mx_idx = 0;
        for (int k = 0; k < B; k++)
            if (v[k] > mx_val) begin
                mx_val = v[k];
                mx_idx = k[2:0];
            end
        return mx_idx;
    endfunction

    int pass = 0, fail = 0;

    task run_batch(input [W-1:0] vecs [N_VEC][B], input string label);
        logic [2:0] exp_b [N_VEC];
        do_reset();
        for (int t = 0; t < B; t++) begin
            @(negedge d_clk);
            for (int u = 0; u < N_VEC; u++) val[u] = vecs[u][t];
        end
        // sample after last posedge
        @(posedge d_clk); #1;
        for (int u = 0; u < N_VEC; u++) exp_b[u] = golden_argmax(vecs[u]);
        begin
            bit ok = 1;
            for (int u = 0; u < N_VEC; u++) if (bucket[u] !== exp_b[u]) ok = 0;
            if (ok) begin
                pass++;
                $display("PASS  [%s]  b={%0d,%0d,%0d,%0d,%0d}",
                         label, bucket[0], bucket[1], bucket[2], bucket[3], bucket[4]);
            end else begin
                fail++;
                $display("FAIL  [%s]  got={%0d,%0d,%0d,%0d,%0d} exp={%0d,%0d,%0d,%0d,%0d}",
                         label,
                         bucket[0], bucket[1], bucket[2], bucket[3], bucket[4],
                         exp_b[0], exp_b[1], exp_b[2], exp_b[3], exp_b[4]);
            end
        end
    endtask

    // stimulus
    logic [W-1:0] vecs [N_VEC][B];
    initial begin
        $display("========== ArgMax Array Integration TB ==========");

        // deterministic cases
        for (int u = 0; u < N_VEC; u++)
            for (int t = 0; t < B; t++) vecs[u][t] = t[2:0];
        run_batch(vecs, "all ramps");

        for (int u = 0; u < N_VEC; u++)
            for (int t = 0; t < B; t++) vecs[u][t] = 3'd4;
        run_batch(vecs, "all ties -> first wins");

        for (int u = 0; u < N_VEC; u++)
            for (int t = 0; t < B; t++) vecs[u][t] = 3'd0;
        run_batch(vecs, "all zero");

        // random
        for (int trial = 0; trial < 40; trial++) begin
            for (int u = 0; u < N_VEC; u++)
                for (int t = 0; t < B; t++) vecs[u][t] = $urandom_range(0,7);
            run_batch(vecs, $sformatf("rand%0d", trial));
        end

        $display("==================================================");
        $display("Results: %0d passed, %0d failed", pass, fail);
        $finish;
    end

endmodule
