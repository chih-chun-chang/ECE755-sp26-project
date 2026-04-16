/*
    Temporal Sorting Unit for Reformer Accelerator

    Inputs:
        -i_b_i: From ArgMax units, multiple bucket indicies to be sorted
    Outputs:
        -o_index: index of input vector
        -o_bucket: what bucket the input vector has been sorted into
        -o_valid: indicates if the other outputs are valid or not
        
*/
`default_nettype none

module SortingUnit(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_start_sort,
    input  wire [2:0] i_b_i0,
    input  wire [2:0] i_b_i1,
    input  wire [2:0] i_b_i2,
    input  wire [2:0] i_b_i3,
    input  wire [2:0] i_b_i4,
    output wire [2:0] o_index,
    output wire [2:0] o_bucket,
    output wire       o_valid
);

reg [2:0] bucket_reg [0:4];
reg [4:0] valid_reg;
reg [2:0] bucket_counter;
reg       busy;

reg [2:0] out_index_r;
reg [2:0] out_bucket_r;
reg       out_valid_r;

reg [4:0] match;
reg       any_match;
reg       any_valid;
reg [2:0] sel_index;
reg       sel_valid;
reg [4:0] valid_after_clear;

always_comb begin
    match[0] = valid_reg[0] && (bucket_reg[0] == bucket_counter);
    match[1] = valid_reg[1] && (bucket_reg[1] == bucket_counter);
    match[2] = valid_reg[2] && (bucket_reg[2] == bucket_counter);
    match[3] = valid_reg[3] && (bucket_reg[3] == bucket_counter);
    match[4] = valid_reg[4] && (bucket_reg[4] == bucket_counter);

    any_match = |match;
    any_valid = |valid_reg;

    sel_valid = 1'b0;
    sel_index = 3'd0;

    if (match[0]) begin
        sel_valid = 1'b1;
        sel_index = 3'd0;
    end
    else if (match[1]) begin
        sel_valid = 1'b1;
        sel_index = 3'd1;
    end
    else if (match[2]) begin
        sel_valid = 1'b1;
        sel_index = 3'd2;
    end
    else if (match[3]) begin
        sel_valid = 1'b1;
        sel_index = 3'd3;
    end
    else if (match[4]) begin
        sel_valid = 1'b1;
        sel_index = 3'd4;
    end

    valid_after_clear = valid_reg;
    if (sel_valid)
        valid_after_clear[sel_index] = 1'b0;
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bucket_reg[0]   <= 3'b0;
        bucket_reg[1]   <= 3'b0;
        bucket_reg[2]   <= 3'b0;
        bucket_reg[3]   <= 3'b0;
        bucket_reg[4]   <= 3'b0;
        valid_reg       <= 5'b0;
        bucket_counter  <= 3'b0;
        busy            <= 1'b0;
        out_index_r     <= 3'b0;
        out_bucket_r    <= 3'b0;
        out_valid_r     <= 1'b0;
    end
    else begin
        out_valid_r <= 1'b0;
        if (i_start_sort) begin
            bucket_reg[0]  <= i_b_i0;
            bucket_reg[1]  <= i_b_i1;
            bucket_reg[2]  <= i_b_i2;
            bucket_reg[3]  <= i_b_i3;
            bucket_reg[4]  <= i_b_i4;
            valid_reg      <= 5'b11111;
            bucket_counter <= 3'b0;
            busy           <= 1'b1;
        end
        else if (busy) begin
            if (any_match && sel_valid) begin
                out_index_r  <= sel_index;
                out_bucket_r <= bucket_counter;
                out_valid_r  <= 1'b1;
                valid_reg    <= valid_after_clear;

                if (valid_after_clear == 5'b0)
                    busy <= 1'b0;
            end
            else if (any_valid) begin
                bucket_counter <= bucket_counter + 3'd1;
            end
            else begin
                busy <= 1'b0;
            end
        end
    end
end

assign o_index  = out_index_r;
assign o_bucket = out_bucket_r;
assign o_valid  = out_valid_r;

endmodule
