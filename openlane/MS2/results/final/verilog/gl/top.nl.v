// This is the unpowered netlist.
module top (clk,
    i_valid,
    o_valid,
    rst_n,
    i_value0,
    i_value1,
    i_value2,
    i_value3,
    i_value4,
    o_bucket,
    o_index);
 input clk;
 input i_valid;
 output o_valid;
 input rst_n;
 input [2:0] i_value0;
 input [2:0] i_value1;
 input [2:0] i_value2;
 input [2:0] i_value3;
 input [2:0] i_value4;
 output [2:0] o_bucket;
 output [2:0] o_index;

 wire _000_;
 wire _001_;
 wire _002_;
 wire _003_;
 wire _004_;
 wire _005_;
 wire _006_;
 wire _007_;
 wire _008_;
 wire _009_;
 wire _010_;
 wire _011_;
 wire _012_;
 wire _013_;
 wire _014_;
 wire _015_;
 wire _016_;
 wire _017_;
 wire _018_;
 wire _019_;
 wire _020_;
 wire _021_;
 wire _022_;
 wire _023_;
 wire _024_;
 wire _025_;
 wire _026_;
 wire _027_;
 wire _028_;
 wire _029_;
 wire _030_;
 wire _031_;
 wire _032_;
 wire _033_;
 wire _034_;
 wire _035_;
 wire _036_;
 wire _037_;
 wire _038_;
 wire _039_;
 wire _040_;
 wire _041_;
 wire _042_;
 wire _043_;
 wire _044_;
 wire _045_;
 wire _046_;
 wire _047_;
 wire _048_;
 wire _049_;
 wire _050_;
 wire _051_;
 wire _052_;
 wire _053_;
 wire _054_;
 wire _055_;
 wire _056_;
 wire _057_;
 wire _058_;
 wire _059_;
 wire _060_;
 wire _061_;
 wire _062_;
 wire _063_;
 wire _064_;
 wire _065_;
 wire _066_;
 wire _067_;
 wire _068_;
 wire _069_;
 wire _070_;
 wire _071_;
 wire _072_;
 wire _073_;
 wire _074_;
 wire _075_;
 wire _076_;
 wire _077_;
 wire _078_;
 wire _079_;
 wire _080_;
 wire _081_;
 wire _082_;
 wire _083_;
 wire _084_;
 wire _085_;
 wire _086_;
 wire _087_;
 wire _088_;
 wire _089_;
 wire _090_;
 wire _091_;
 wire _092_;
 wire _093_;
 wire _094_;
 wire _095_;
 wire _096_;
 wire _097_;
 wire _098_;
 wire _099_;
 wire _100_;
 wire _101_;
 wire _102_;
 wire _103_;
 wire _104_;
 wire _105_;
 wire _106_;
 wire _107_;
 wire _108_;
 wire _109_;
 wire _110_;
 wire _111_;
 wire _112_;
 wire _113_;
 wire _114_;
 wire _115_;
 wire _116_;
 wire _117_;
 wire _118_;
 wire _119_;
 wire _120_;
 wire _121_;
 wire _122_;
 wire _123_;
 wire _124_;
 wire _125_;
 wire _126_;
 wire _127_;
 wire _128_;
 wire _129_;
 wire _130_;
 wire _131_;
 wire _132_;
 wire _133_;
 wire _134_;
 wire _135_;
 wire _136_;
 wire _137_;
 wire _138_;
 wire _139_;
 wire _140_;
 wire _141_;
 wire _142_;
 wire _143_;
 wire _144_;
 wire _145_;
 wire _146_;
 wire _147_;
 wire _148_;
 wire _149_;
 wire _150_;
 wire _151_;
 wire _152_;
 wire _153_;
 wire _154_;
 wire _155_;
 wire _156_;
 wire _157_;
 wire _158_;
 wire _159_;
 wire _160_;
 wire _161_;
 wire _162_;
 wire _163_;
 wire _164_;
 wire _165_;
 wire _166_;
 wire _167_;
 wire _168_;
 wire _169_;
 wire _170_;
 wire _171_;
 wire _172_;
 wire _173_;
 wire _174_;
 wire _175_;
 wire _176_;
 wire _177_;
 wire _178_;
 wire _179_;
 wire _180_;
 wire _181_;
 wire _182_;
 wire _183_;
 wire _184_;
 wire _185_;
 wire _186_;
 wire _187_;
 wire _188_;
 wire _189_;
 wire _190_;
 wire _191_;
 wire _192_;
 wire _193_;
 wire _194_;
 wire _195_;
 wire _196_;
 wire _197_;
 wire _198_;
 wire _199_;
 wire _200_;
 wire _201_;
 wire _202_;
 wire _203_;
 wire _204_;
 wire _205_;
 wire _206_;
 wire _207_;
 wire _208_;
 wire _209_;
 wire _210_;
 wire clknet_0_clk;
 wire clknet_3_0__leaf_clk;
 wire clknet_3_1__leaf_clk;
 wire clknet_3_2__leaf_clk;
 wire clknet_3_3__leaf_clk;
 wire clknet_3_4__leaf_clk;
 wire clknet_3_5__leaf_clk;
 wire clknet_3_6__leaf_clk;
 wire clknet_3_7__leaf_clk;
 wire net1;
 wire net10;
 wire net11;
 wire net12;
 wire net13;
 wire net14;
 wire net15;
 wire net16;
 wire net17;
 wire net18;
 wire net19;
 wire net2;
 wire net20;
 wire net21;
 wire net22;
 wire net23;
 wire net24;
 wire net25;
 wire net26;
 wire net27;
 wire net28;
 wire net29;
 wire net3;
 wire net30;
 wire net31;
 wire net32;
 wire net33;
 wire net34;
 wire net35;
 wire net36;
 wire net37;
 wire net38;
 wire net39;
 wire net4;
 wire net40;
 wire net41;
 wire net42;
 wire net43;
 wire net44;
 wire net45;
 wire net46;
 wire net47;
 wire net48;
 wire net49;
 wire net5;
 wire net50;
 wire net51;
 wire net52;
 wire net53;
 wire net54;
 wire net55;
 wire net56;
 wire net57;
 wire net58;
 wire net59;
 wire net6;
 wire net60;
 wire net61;
 wire net62;
 wire net63;
 wire net64;
 wire net65;
 wire net66;
 wire net67;
 wire net68;
 wire net69;
 wire net7;
 wire net70;
 wire net71;
 wire net72;
 wire net73;
 wire net8;
 wire net9;
 wire \u_am0.b_i[0] ;
 wire \u_am0.b_i[1] ;
 wire \u_am0.b_i[2] ;
 wire \u_am0.i_index[0] ;
 wire \u_am0.i_index[1] ;
 wire \u_am0.i_index[2] ;
 wire \u_am0.max[0] ;
 wire \u_am0.max[1] ;
 wire \u_am0.max[2] ;
 wire \u_am1.b_i[0] ;
 wire \u_am1.b_i[1] ;
 wire \u_am1.b_i[2] ;
 wire \u_am1.max[0] ;
 wire \u_am1.max[1] ;
 wire \u_am1.max[2] ;
 wire \u_am2.b_i[0] ;
 wire \u_am2.b_i[1] ;
 wire \u_am2.b_i[2] ;
 wire \u_am2.max[0] ;
 wire \u_am2.max[1] ;
 wire \u_am2.max[2] ;
 wire \u_am3.b_i[0] ;
 wire \u_am3.b_i[1] ;
 wire \u_am3.b_i[2] ;
 wire \u_am3.max[0] ;
 wire \u_am3.max[1] ;
 wire \u_am3.max[2] ;
 wire \u_am4.b_i[0] ;
 wire \u_am4.b_i[1] ;
 wire \u_am4.b_i[2] ;
 wire \u_am4.max[0] ;
 wire \u_am4.max[1] ;
 wire \u_am4.max[2] ;
 wire \u_control.o_start_sort ;
 wire \u_sort.bucket_counter[0] ;
 wire \u_sort.bucket_counter[1] ;
 wire \u_sort.bucket_counter[2] ;
 wire \u_sort.bucket_reg[0][0] ;
 wire \u_sort.bucket_reg[0][1] ;
 wire \u_sort.bucket_reg[0][2] ;
 wire \u_sort.bucket_reg[1][0] ;
 wire \u_sort.bucket_reg[1][1] ;
 wire \u_sort.bucket_reg[1][2] ;
 wire \u_sort.bucket_reg[2][0] ;
 wire \u_sort.bucket_reg[2][1] ;
 wire \u_sort.bucket_reg[2][2] ;
 wire \u_sort.bucket_reg[3][0] ;
 wire \u_sort.bucket_reg[3][1] ;
 wire \u_sort.bucket_reg[3][2] ;
 wire \u_sort.bucket_reg[4][0] ;
 wire \u_sort.bucket_reg[4][1] ;
 wire \u_sort.bucket_reg[4][2] ;
 wire \u_sort.busy ;
 wire \u_sort.valid_reg[0] ;
 wire \u_sort.valid_reg[1] ;
 wire \u_sort.valid_reg[2] ;
 wire \u_sort.valid_reg[3] ;
 wire \u_sort.valid_reg[4] ;

 sky130_fd_sc_hd__decap_8 FILLER_0_0_102 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_110 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_0_119 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_0_123 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_138 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_0_144 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_0_156 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_0_160 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_0_169 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_0_23 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_0_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_0_29 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_45 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_0_6 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_0_60 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_0_64 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_0_73 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_82 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_85 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_0_90 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_10_102 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_10_110 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_119 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_10_131 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_10_139 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_10_141 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_10_145 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_154 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_10_166 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_10_174 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_10_23 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_70 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_10_82 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_11_110 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_11_113 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_11_121 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_11_126 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_11_134 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_11_15 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_11_162 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_11_169 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_3 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_11_48 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_57 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_11_69 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_11_77 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_98 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_12_134 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_12_139 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_12_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_12_161 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_12_173 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_12_27 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_12_29 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_12_3 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_12_71 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_12_82 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_12_85 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_12_91 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_12_97 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_13_111 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_13_130 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_13_138 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_13_15 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_13_159 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_13_167 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_13_169 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_13_19 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_13_3 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_13_40 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_13_46 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_13_82 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_13_99 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_14_100 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_14_134 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_14_141 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_14_145 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_14_157 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_14_38 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_14_50 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_14_6 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_14_62 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_14_69 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_14_85 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_15_105 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_15_136 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_15_148 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_15_15 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_15_160 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_15_169 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_15_19 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_15_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_15_55 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_15_62 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_15_72 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_16_100 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_16_112 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_16_124 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_16_136 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_16_141 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_16_15 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_16_165 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_16_173 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_16_23 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_16_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_16_29 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_16_3 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_16_41 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_16_49 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_16_74 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_16_82 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_113 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_17_125 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_17_158 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_17_167 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_17_169 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_20 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_17_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_32 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_44 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_60 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_17_7 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_17_72 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_17_84 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_18_101 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_18_120 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_128 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_18_138 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_18_156 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_18_165 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_23 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_18_60 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_68 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_18_79 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_83 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_18_85 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_18_93 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_19_169 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_19_3 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_19_57 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_19_69 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_19_7 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_19_77 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_19_81 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_1_133 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_1_154 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_1_166 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_1_169 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_1_37 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_1_54 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_1_66 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_20_121 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_20_133 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_139 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_17 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_170 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_20_21 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_27 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_20_38 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_44 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_20_83 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_20_89 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_20_9 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_20_99 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_21_103 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_21_11 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_21_113 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_21_118 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_21_129 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_21_150 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_21_164 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_21_169 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_21_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_21_41 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_21_51 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_21_55 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_21_63 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_21_69 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_21_86 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_21_90 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_21_95 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_22_128 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_22_136 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_22_15 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_22_150 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_22_161 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_22_173 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_22_23 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_22_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_22_49 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_22_56 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_22_85 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_23_102 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_23_110 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_23_113 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_23_122 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_23_14 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_23_161 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_23_167 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_23_169 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_23_24 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_23_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_23_65 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_23_77 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_23_86 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_23_98 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_24_105 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_24_133 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_24_139 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_24_150 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_24_25 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_24_3 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_24_37 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_24_70 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_24_82 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_24_85 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_24_93 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_25_106 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_25_113 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_25_121 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_25_129 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_25_167 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_25_169 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_25_21 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_25_33 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_25_43 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_25_48 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_25_52 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_25_77 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_25_9 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_118 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_26_130 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_26_134 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_26_138 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_26_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_152 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_26_164 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_26_23 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_57 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_69 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_26_81 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_26_85 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_26_97 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_27_113 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_27_121 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_27_127 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_27_141 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_27_15 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_27_164 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_27_169 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_27_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_27_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_27_31 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_27_52 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_27_66 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_27_78 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_27_82 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_28_105 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_28_137 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_28_141 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_28_147 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_28_168 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_28_174 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_28_24 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_28_3 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_29_100 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_29_107 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_111 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_29_113 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_29_12 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_29_125 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_29_133 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_139 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_149 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_29_156 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_29_24 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_29_29 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_37 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_29_41 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_29_51 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_55 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_64 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_29_82 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_29_85 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_29_91 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_2_106 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_2_114 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_2_120 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_2_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_2_161 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_2_173 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_2_23 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_2_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_2_69 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_2_79 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_2_95 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_3_141 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_15 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_3_151 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_3_161 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_3_167 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_3_169 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_3_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_42 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_3_54 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_3_57 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_3_83 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_4_11 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_4_123 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_4_131 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_4_136 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_4_141 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_4_149 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_4_154 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_4_166 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_4_174 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_4_19 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_4_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_4_37 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_4_55 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_4_79 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_4_83 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_4_85 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_4_91 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_5_122 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_5_156 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_5_169 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_5_26 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_5_3 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_5_49 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_5_55 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_5_57 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_5_65 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_5_79 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_5_91 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_100 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_121 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_6_131 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_139 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_6_141 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_6_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_6_153 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_6_165 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_6_173 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_19 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_6_3 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_6_32 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_40 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_6_94 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_7_101 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_7_105 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_7_113 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_7_12 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_7_142 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_7_148 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_7_160 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_7_169 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_7_20 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_7_38 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_7_44 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_7_54 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_7_57 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_8_159 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_8_169 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_8_26 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_8_40 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_8_72 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_8_96 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_9_105 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_9_11 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_9_111 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_9_113 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_9_125 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_9_167 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_9_169 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_9_3 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_9_53 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_9_72 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_9_81 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_9_93 ();
 sky130_fd_sc_hd__decap_3 PHY_0 ();
 sky130_fd_sc_hd__decap_3 PHY_1 ();
 sky130_fd_sc_hd__decap_3 PHY_10 ();
 sky130_fd_sc_hd__decap_3 PHY_11 ();
 sky130_fd_sc_hd__decap_3 PHY_12 ();
 sky130_fd_sc_hd__decap_3 PHY_13 ();
 sky130_fd_sc_hd__decap_3 PHY_14 ();
 sky130_fd_sc_hd__decap_3 PHY_15 ();
 sky130_fd_sc_hd__decap_3 PHY_16 ();
 sky130_fd_sc_hd__decap_3 PHY_17 ();
 sky130_fd_sc_hd__decap_3 PHY_18 ();
 sky130_fd_sc_hd__decap_3 PHY_19 ();
 sky130_fd_sc_hd__decap_3 PHY_2 ();
 sky130_fd_sc_hd__decap_3 PHY_20 ();
 sky130_fd_sc_hd__decap_3 PHY_21 ();
 sky130_fd_sc_hd__decap_3 PHY_22 ();
 sky130_fd_sc_hd__decap_3 PHY_23 ();
 sky130_fd_sc_hd__decap_3 PHY_24 ();
 sky130_fd_sc_hd__decap_3 PHY_25 ();
 sky130_fd_sc_hd__decap_3 PHY_26 ();
 sky130_fd_sc_hd__decap_3 PHY_27 ();
 sky130_fd_sc_hd__decap_3 PHY_28 ();
 sky130_fd_sc_hd__decap_3 PHY_29 ();
 sky130_fd_sc_hd__decap_3 PHY_3 ();
 sky130_fd_sc_hd__decap_3 PHY_30 ();
 sky130_fd_sc_hd__decap_3 PHY_31 ();
 sky130_fd_sc_hd__decap_3 PHY_32 ();
 sky130_fd_sc_hd__decap_3 PHY_33 ();
 sky130_fd_sc_hd__decap_3 PHY_34 ();
 sky130_fd_sc_hd__decap_3 PHY_35 ();
 sky130_fd_sc_hd__decap_3 PHY_36 ();
 sky130_fd_sc_hd__decap_3 PHY_37 ();
 sky130_fd_sc_hd__decap_3 PHY_38 ();
 sky130_fd_sc_hd__decap_3 PHY_39 ();
 sky130_fd_sc_hd__decap_3 PHY_4 ();
 sky130_fd_sc_hd__decap_3 PHY_40 ();
 sky130_fd_sc_hd__decap_3 PHY_41 ();
 sky130_fd_sc_hd__decap_3 PHY_42 ();
 sky130_fd_sc_hd__decap_3 PHY_43 ();
 sky130_fd_sc_hd__decap_3 PHY_44 ();
 sky130_fd_sc_hd__decap_3 PHY_45 ();
 sky130_fd_sc_hd__decap_3 PHY_46 ();
 sky130_fd_sc_hd__decap_3 PHY_47 ();
 sky130_fd_sc_hd__decap_3 PHY_48 ();
 sky130_fd_sc_hd__decap_3 PHY_49 ();
 sky130_fd_sc_hd__decap_3 PHY_5 ();
 sky130_fd_sc_hd__decap_3 PHY_50 ();
 sky130_fd_sc_hd__decap_3 PHY_51 ();
 sky130_fd_sc_hd__decap_3 PHY_52 ();
 sky130_fd_sc_hd__decap_3 PHY_53 ();
 sky130_fd_sc_hd__decap_3 PHY_54 ();
 sky130_fd_sc_hd__decap_3 PHY_55 ();
 sky130_fd_sc_hd__decap_3 PHY_56 ();
 sky130_fd_sc_hd__decap_3 PHY_57 ();
 sky130_fd_sc_hd__decap_3 PHY_58 ();
 sky130_fd_sc_hd__decap_3 PHY_59 ();
 sky130_fd_sc_hd__decap_3 PHY_6 ();
 sky130_fd_sc_hd__decap_3 PHY_7 ();
 sky130_fd_sc_hd__decap_3 PHY_8 ();
 sky130_fd_sc_hd__decap_3 PHY_9 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_100 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_101 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_102 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_103 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_104 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_105 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_106 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_107 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_108 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_109 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_110 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_111 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_112 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_113 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_114 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_115 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_116 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_117 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_118 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_119 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_120 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_121 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_122 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_123 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_124 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_125 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_126 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_127 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_128 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_129 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_130 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_131 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_132 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_133 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_134 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_135 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_136 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_137 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_138 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_139 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_140 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_141 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_142 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_143 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_144 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_145 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_146 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_147 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_148 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_149 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_150 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_151 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_152 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_153 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_154 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_155 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_60 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_61 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_62 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_63 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_64 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_65 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_66 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_67 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_68 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_69 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_70 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_71 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_72 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_73 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_74 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_75 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_76 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_77 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_78 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_79 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_80 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_81 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_82 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_83 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_84 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_85 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_86 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_87 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_88 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_89 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_90 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_91 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_92 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_93 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_94 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_95 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_96 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_97 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_98 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_99 ();
 sky130_fd_sc_hd__buf_4 _211_ (.A(net1),
    .X(_065_));
 sky130_fd_sc_hd__and3_1 _212_ (.A(_065_),
    .B(\u_am0.i_index[1] ),
    .C(\u_am0.i_index[0] ),
    .X(_066_));
 sky130_fd_sc_hd__and2_1 _213_ (.A(\u_am0.i_index[2] ),
    .B(_066_),
    .X(_067_));
 sky130_fd_sc_hd__clkbuf_1 _214_ (.A(_067_),
    .X(_000_));
 sky130_fd_sc_hd__inv_2 _215_ (.A(\u_sort.bucket_reg[4][0] ),
    .Y(_068_));
 sky130_fd_sc_hd__inv_2 _216_ (.A(\u_sort.valid_reg[4] ),
    .Y(_069_));
 sky130_fd_sc_hd__inv_2 _217_ (.A(\u_sort.bucket_counter[0] ),
    .Y(_070_));
 sky130_fd_sc_hd__clkbuf_4 _218_ (.A(\u_sort.bucket_counter[1] ),
    .X(_071_));
 sky130_fd_sc_hd__or2_1 _219_ (.A(_071_),
    .B(\u_sort.bucket_reg[4][1] ),
    .X(_072_));
 sky130_fd_sc_hd__nand2_1 _220_ (.A(_071_),
    .B(\u_sort.bucket_reg[4][1] ),
    .Y(_073_));
 sky130_fd_sc_hd__clkbuf_4 _221_ (.A(\u_sort.bucket_counter[2] ),
    .X(_074_));
 sky130_fd_sc_hd__xor2_1 _222_ (.A(_074_),
    .B(\u_sort.bucket_reg[4][2] ),
    .X(_075_));
 sky130_fd_sc_hd__a221o_1 _223_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[4][0] ),
    .B1(_072_),
    .B2(_073_),
    .C1(_075_),
    .X(_076_));
 sky130_fd_sc_hd__a211o_1 _224_ (.A1(\u_sort.bucket_counter[0] ),
    .A2(_068_),
    .B1(_069_),
    .C1(_076_),
    .X(_077_));
 sky130_fd_sc_hd__inv_2 _225_ (.A(_071_),
    .Y(_078_));
 sky130_fd_sc_hd__and2b_1 _226_ (.A_N(\u_sort.bucket_reg[1][2] ),
    .B(_074_),
    .X(_079_));
 sky130_fd_sc_hd__a221oi_2 _227_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[1][0] ),
    .B1(\u_sort.bucket_reg[1][1] ),
    .B2(_078_),
    .C1(_079_),
    .Y(_080_));
 sky130_fd_sc_hd__or2b_1 _228_ (.A(\u_sort.bucket_reg[1][0] ),
    .B_N(\u_sort.bucket_counter[0] ),
    .X(_081_));
 sky130_fd_sc_hd__or2b_1 _229_ (.A(_074_),
    .B_N(\u_sort.bucket_reg[1][2] ),
    .X(_082_));
 sky130_fd_sc_hd__o2111a_1 _230_ (.A1(_078_),
    .A2(\u_sort.bucket_reg[1][1] ),
    .B1(\u_sort.valid_reg[1] ),
    .C1(_081_),
    .D1(_082_),
    .X(_083_));
 sky130_fd_sc_hd__xnor2_1 _231_ (.A(\u_sort.bucket_reg[0][2] ),
    .B(_074_),
    .Y(_084_));
 sky130_fd_sc_hd__xnor2_1 _232_ (.A(\u_sort.bucket_reg[0][1] ),
    .B(_071_),
    .Y(_085_));
 sky130_fd_sc_hd__xnor2_1 _233_ (.A(\u_sort.bucket_reg[0][0] ),
    .B(\u_sort.bucket_counter[0] ),
    .Y(_086_));
 sky130_fd_sc_hd__and4_1 _234_ (.A(\u_sort.valid_reg[0] ),
    .B(_084_),
    .C(_085_),
    .D(_086_),
    .X(_087_));
 sky130_fd_sc_hd__a21oi_1 _235_ (.A1(_080_),
    .A2(_083_),
    .B1(_087_),
    .Y(_088_));
 sky130_fd_sc_hd__o21ai_1 _236_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[3][0] ),
    .B1(\u_sort.valid_reg[3] ),
    .Y(_089_));
 sky130_fd_sc_hd__xor2_1 _237_ (.A(_071_),
    .B(\u_sort.bucket_reg[3][1] ),
    .X(_090_));
 sky130_fd_sc_hd__xor2_1 _238_ (.A(_074_),
    .B(\u_sort.bucket_reg[3][2] ),
    .X(_091_));
 sky130_fd_sc_hd__a2111o_1 _239_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[3][0] ),
    .B1(_089_),
    .C1(_090_),
    .D1(_091_),
    .X(_092_));
 sky130_fd_sc_hd__o21ai_1 _240_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[2][0] ),
    .B1(\u_sort.valid_reg[2] ),
    .Y(_093_));
 sky130_fd_sc_hd__xor2_1 _241_ (.A(_071_),
    .B(\u_sort.bucket_reg[2][1] ),
    .X(_094_));
 sky130_fd_sc_hd__xor2_1 _242_ (.A(_074_),
    .B(\u_sort.bucket_reg[2][2] ),
    .X(_095_));
 sky130_fd_sc_hd__a2111o_2 _243_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[2][0] ),
    .B1(_093_),
    .C1(_094_),
    .D1(_095_),
    .X(_096_));
 sky130_fd_sc_hd__and4_1 _244_ (.A(_077_),
    .B(_088_),
    .C(_092_),
    .D(_096_),
    .X(_097_));
 sky130_fd_sc_hd__buf_6 _245_ (.A(\u_control.o_start_sort ),
    .X(_098_));
 sky130_fd_sc_hd__inv_2 _246_ (.A(_098_),
    .Y(_099_));
 sky130_fd_sc_hd__and3b_1 _247_ (.A_N(_097_),
    .B(\u_sort.busy ),
    .C(_099_),
    .X(_100_));
 sky130_fd_sc_hd__clkbuf_4 _248_ (.A(_100_),
    .X(_002_));
 sky130_fd_sc_hd__inv_2 _249_ (.A(\u_sort.busy ),
    .Y(_101_));
 sky130_fd_sc_hd__and4b_1 _250_ (.A_N(_077_),
    .B(_088_),
    .C(_092_),
    .D(_096_),
    .X(_102_));
 sky130_fd_sc_hd__a2111oi_1 _251_ (.A1(_070_),
    .A2(\u_sort.bucket_reg[3][0] ),
    .B1(_089_),
    .C1(_090_),
    .D1(_091_),
    .Y(_103_));
 sky130_fd_sc_hd__inv_2 _252_ (.A(\u_sort.valid_reg[3] ),
    .Y(_104_));
 sky130_fd_sc_hd__a31o_1 _253_ (.A1(_088_),
    .A2(net25),
    .A3(_096_),
    .B1(_104_),
    .X(_105_));
 sky130_fd_sc_hd__nand3_1 _254_ (.A(_084_),
    .B(_085_),
    .C(_086_),
    .Y(_106_));
 sky130_fd_sc_hd__nand3b_1 _255_ (.A_N(_087_),
    .B(_080_),
    .C(_083_),
    .Y(_107_));
 sky130_fd_sc_hd__a22oi_1 _256_ (.A1(\u_sort.valid_reg[0] ),
    .A2(_106_),
    .B1(_107_),
    .B2(\u_sort.valid_reg[1] ),
    .Y(_108_));
 sky130_fd_sc_hd__a21o_1 _257_ (.A1(_080_),
    .A2(_083_),
    .B1(_087_),
    .X(_109_));
 sky130_fd_sc_hd__o21ai_1 _258_ (.A1(_109_),
    .A2(_096_),
    .B1(\u_sort.valid_reg[2] ),
    .Y(_110_));
 sky130_fd_sc_hd__o2111a_2 _259_ (.A1(_069_),
    .A2(_102_),
    .B1(_105_),
    .C1(_108_),
    .D1(_110_),
    .X(_111_));
 sky130_fd_sc_hd__o21ai_1 _260_ (.A1(_101_),
    .A2(_111_),
    .B1(_099_),
    .Y(_001_));
 sky130_fd_sc_hd__inv_2 _261_ (.A(\u_am4.max[2] ),
    .Y(_112_));
 sky130_fd_sc_hd__inv_2 _262_ (.A(\u_am4.max[1] ),
    .Y(_113_));
 sky130_fd_sc_hd__inv_2 _263_ (.A(\u_am4.max[0] ),
    .Y(_114_));
 sky130_fd_sc_hd__o211a_1 _264_ (.A1(net15),
    .A2(_113_),
    .B1(net14),
    .C1(_114_),
    .X(_115_));
 sky130_fd_sc_hd__a22o_1 _265_ (.A1(_112_),
    .A2(net16),
    .B1(net15),
    .B2(_113_),
    .X(_116_));
 sky130_fd_sc_hd__o221a_2 _266_ (.A1(_112_),
    .A2(net16),
    .B1(_115_),
    .B2(_116_),
    .C1(_065_),
    .X(_117_));
 sky130_fd_sc_hd__mux2_1 _267_ (.A0(net57),
    .A1(\u_am0.i_index[0] ),
    .S(_117_),
    .X(_118_));
 sky130_fd_sc_hd__clkbuf_1 _268_ (.A(_118_),
    .X(_003_));
 sky130_fd_sc_hd__mux2_1 _269_ (.A0(net47),
    .A1(\u_am0.i_index[1] ),
    .S(_117_),
    .X(_119_));
 sky130_fd_sc_hd__clkbuf_1 _270_ (.A(_119_),
    .X(_004_));
 sky130_fd_sc_hd__mux2_1 _271_ (.A0(net50),
    .A1(\u_am0.i_index[2] ),
    .S(_117_),
    .X(_120_));
 sky130_fd_sc_hd__clkbuf_1 _272_ (.A(_120_),
    .X(_005_));
 sky130_fd_sc_hd__inv_2 _273_ (.A(\u_am0.max[2] ),
    .Y(_121_));
 sky130_fd_sc_hd__inv_2 _274_ (.A(\u_am0.max[1] ),
    .Y(_122_));
 sky130_fd_sc_hd__inv_2 _275_ (.A(\u_am0.max[0] ),
    .Y(_123_));
 sky130_fd_sc_hd__o211a_1 _276_ (.A1(_122_),
    .A2(net3),
    .B1(net2),
    .C1(_123_),
    .X(_124_));
 sky130_fd_sc_hd__a22o_1 _277_ (.A1(_121_),
    .A2(net4),
    .B1(_122_),
    .B2(net3),
    .X(_125_));
 sky130_fd_sc_hd__o221a_2 _278_ (.A1(_121_),
    .A2(net4),
    .B1(_124_),
    .B2(_125_),
    .C1(_065_),
    .X(_126_));
 sky130_fd_sc_hd__mux2_1 _279_ (.A0(net68),
    .A1(net2),
    .S(_126_),
    .X(_127_));
 sky130_fd_sc_hd__clkbuf_1 _280_ (.A(_127_),
    .X(_006_));
 sky130_fd_sc_hd__mux2_1 _281_ (.A0(net73),
    .A1(net3),
    .S(_126_),
    .X(_128_));
 sky130_fd_sc_hd__clkbuf_1 _282_ (.A(_128_),
    .X(_007_));
 sky130_fd_sc_hd__a21o_1 _283_ (.A1(_065_),
    .A2(net4),
    .B1(net30),
    .X(_008_));
 sky130_fd_sc_hd__inv_2 _284_ (.A(\u_am1.max[2] ),
    .Y(_129_));
 sky130_fd_sc_hd__inv_2 _285_ (.A(\u_am1.max[1] ),
    .Y(_130_));
 sky130_fd_sc_hd__inv_2 _286_ (.A(\u_am1.max[0] ),
    .Y(_131_));
 sky130_fd_sc_hd__o211a_1 _287_ (.A1(_130_),
    .A2(net6),
    .B1(net5),
    .C1(_131_),
    .X(_132_));
 sky130_fd_sc_hd__a22o_1 _288_ (.A1(_129_),
    .A2(net7),
    .B1(_130_),
    .B2(net6),
    .X(_133_));
 sky130_fd_sc_hd__o221a_2 _289_ (.A1(_129_),
    .A2(net7),
    .B1(_132_),
    .B2(_133_),
    .C1(_065_),
    .X(_134_));
 sky130_fd_sc_hd__mux2_1 _290_ (.A0(net67),
    .A1(net5),
    .S(_134_),
    .X(_135_));
 sky130_fd_sc_hd__clkbuf_1 _291_ (.A(_135_),
    .X(_009_));
 sky130_fd_sc_hd__mux2_1 _292_ (.A0(net70),
    .A1(net6),
    .S(_134_),
    .X(_136_));
 sky130_fd_sc_hd__clkbuf_1 _293_ (.A(_136_),
    .X(_010_));
 sky130_fd_sc_hd__a21o_1 _294_ (.A1(_065_),
    .A2(net7),
    .B1(net27),
    .X(_011_));
 sky130_fd_sc_hd__mux2_1 _295_ (.A0(net59),
    .A1(\u_am0.i_index[0] ),
    .S(_126_),
    .X(_137_));
 sky130_fd_sc_hd__clkbuf_1 _296_ (.A(_137_),
    .X(_012_));
 sky130_fd_sc_hd__mux2_1 _297_ (.A0(net56),
    .A1(\u_am0.i_index[1] ),
    .S(_126_),
    .X(_138_));
 sky130_fd_sc_hd__clkbuf_1 _298_ (.A(_138_),
    .X(_013_));
 sky130_fd_sc_hd__mux2_1 _299_ (.A0(net51),
    .A1(\u_am0.i_index[2] ),
    .S(_126_),
    .X(_139_));
 sky130_fd_sc_hd__clkbuf_1 _300_ (.A(_139_),
    .X(_014_));
 sky130_fd_sc_hd__inv_2 _301_ (.A(\u_am2.max[2] ),
    .Y(_140_));
 sky130_fd_sc_hd__inv_2 _302_ (.A(\u_am2.max[1] ),
    .Y(_141_));
 sky130_fd_sc_hd__inv_2 _303_ (.A(\u_am2.max[0] ),
    .Y(_142_));
 sky130_fd_sc_hd__o211a_1 _304_ (.A1(_141_),
    .A2(net9),
    .B1(net8),
    .C1(_142_),
    .X(_143_));
 sky130_fd_sc_hd__a22o_1 _305_ (.A1(_140_),
    .A2(net10),
    .B1(_141_),
    .B2(net9),
    .X(_144_));
 sky130_fd_sc_hd__o221a_2 _306_ (.A1(_140_),
    .A2(net10),
    .B1(_143_),
    .B2(_144_),
    .C1(_065_),
    .X(_145_));
 sky130_fd_sc_hd__mux2_1 _307_ (.A0(net64),
    .A1(net8),
    .S(_145_),
    .X(_146_));
 sky130_fd_sc_hd__clkbuf_1 _308_ (.A(_146_),
    .X(_015_));
 sky130_fd_sc_hd__mux2_1 _309_ (.A0(net71),
    .A1(net9),
    .S(_145_),
    .X(_147_));
 sky130_fd_sc_hd__clkbuf_1 _310_ (.A(_147_),
    .X(_016_));
 sky130_fd_sc_hd__a21o_1 _311_ (.A1(_065_),
    .A2(net10),
    .B1(net31),
    .X(_017_));
 sky130_fd_sc_hd__mux2_1 _312_ (.A0(net42),
    .A1(\u_am0.i_index[0] ),
    .S(_134_),
    .X(_148_));
 sky130_fd_sc_hd__clkbuf_1 _313_ (.A(_148_),
    .X(_018_));
 sky130_fd_sc_hd__mux2_1 _314_ (.A0(net53),
    .A1(\u_am0.i_index[1] ),
    .S(_134_),
    .X(_149_));
 sky130_fd_sc_hd__clkbuf_1 _315_ (.A(_149_),
    .X(_019_));
 sky130_fd_sc_hd__mux2_1 _316_ (.A0(net38),
    .A1(\u_am0.i_index[2] ),
    .S(_134_),
    .X(_150_));
 sky130_fd_sc_hd__clkbuf_1 _317_ (.A(_150_),
    .X(_020_));
 sky130_fd_sc_hd__inv_2 _318_ (.A(\u_am3.max[2] ),
    .Y(_151_));
 sky130_fd_sc_hd__inv_2 _319_ (.A(\u_am3.max[1] ),
    .Y(_152_));
 sky130_fd_sc_hd__inv_2 _320_ (.A(\u_am3.max[0] ),
    .Y(_153_));
 sky130_fd_sc_hd__o211a_1 _321_ (.A1(net12),
    .A2(_152_),
    .B1(net11),
    .C1(_153_),
    .X(_154_));
 sky130_fd_sc_hd__a22o_1 _322_ (.A1(_151_),
    .A2(net13),
    .B1(net12),
    .B2(_152_),
    .X(_155_));
 sky130_fd_sc_hd__o221a_2 _323_ (.A1(_151_),
    .A2(net13),
    .B1(_154_),
    .B2(_155_),
    .C1(_065_),
    .X(_156_));
 sky130_fd_sc_hd__mux2_1 _324_ (.A0(net69),
    .A1(net11),
    .S(_156_),
    .X(_157_));
 sky130_fd_sc_hd__clkbuf_1 _325_ (.A(_157_),
    .X(_021_));
 sky130_fd_sc_hd__mux2_1 _326_ (.A0(\u_am3.max[1] ),
    .A1(net12),
    .S(_156_),
    .X(_158_));
 sky130_fd_sc_hd__clkbuf_1 _327_ (.A(_158_),
    .X(_022_));
 sky130_fd_sc_hd__a21o_1 _328_ (.A1(_065_),
    .A2(net13),
    .B1(net29),
    .X(_023_));
 sky130_fd_sc_hd__mux2_1 _329_ (.A0(net37),
    .A1(\u_am0.i_index[0] ),
    .S(_145_),
    .X(_159_));
 sky130_fd_sc_hd__clkbuf_1 _330_ (.A(_159_),
    .X(_024_));
 sky130_fd_sc_hd__mux2_1 _331_ (.A0(net34),
    .A1(\u_am0.i_index[1] ),
    .S(_145_),
    .X(_160_));
 sky130_fd_sc_hd__clkbuf_1 _332_ (.A(_160_),
    .X(_025_));
 sky130_fd_sc_hd__mux2_1 _333_ (.A0(net36),
    .A1(\u_am0.i_index[2] ),
    .S(_145_),
    .X(_161_));
 sky130_fd_sc_hd__clkbuf_1 _334_ (.A(_161_),
    .X(_026_));
 sky130_fd_sc_hd__mux2_1 _335_ (.A0(net65),
    .A1(net14),
    .S(_117_),
    .X(_162_));
 sky130_fd_sc_hd__clkbuf_1 _336_ (.A(_162_),
    .X(_027_));
 sky130_fd_sc_hd__mux2_1 _337_ (.A0(net72),
    .A1(net15),
    .S(_117_),
    .X(_163_));
 sky130_fd_sc_hd__clkbuf_1 _338_ (.A(_163_),
    .X(_028_));
 sky130_fd_sc_hd__a21o_1 _339_ (.A1(net16),
    .A2(_065_),
    .B1(net28),
    .X(_029_));
 sky130_fd_sc_hd__mux2_1 _340_ (.A0(net58),
    .A1(\u_am0.i_index[0] ),
    .S(_156_),
    .X(_164_));
 sky130_fd_sc_hd__clkbuf_1 _341_ (.A(_164_),
    .X(_030_));
 sky130_fd_sc_hd__mux2_1 _342_ (.A0(net45),
    .A1(\u_am0.i_index[1] ),
    .S(_156_),
    .X(_165_));
 sky130_fd_sc_hd__clkbuf_1 _343_ (.A(_165_),
    .X(_031_));
 sky130_fd_sc_hd__mux2_1 _344_ (.A0(net49),
    .A1(\u_am0.i_index[2] ),
    .S(_156_),
    .X(_166_));
 sky130_fd_sc_hd__clkbuf_1 _345_ (.A(_166_),
    .X(_032_));
 sky130_fd_sc_hd__mux2_1 _346_ (.A0(\u_sort.bucket_reg[4][0] ),
    .A1(net57),
    .S(_098_),
    .X(_167_));
 sky130_fd_sc_hd__clkbuf_1 _347_ (.A(_167_),
    .X(_033_));
 sky130_fd_sc_hd__mux2_1 _348_ (.A0(net48),
    .A1(net47),
    .S(_098_),
    .X(_168_));
 sky130_fd_sc_hd__clkbuf_1 _349_ (.A(_168_),
    .X(_034_));
 sky130_fd_sc_hd__mux2_1 _350_ (.A0(net54),
    .A1(net50),
    .S(_098_),
    .X(_169_));
 sky130_fd_sc_hd__clkbuf_1 _351_ (.A(_169_),
    .X(_035_));
 sky130_fd_sc_hd__mux2_1 _352_ (.A0(\u_sort.bucket_reg[3][0] ),
    .A1(net58),
    .S(_098_),
    .X(_170_));
 sky130_fd_sc_hd__clkbuf_1 _353_ (.A(_170_),
    .X(_036_));
 sky130_fd_sc_hd__mux2_1 _354_ (.A0(net52),
    .A1(net45),
    .S(_098_),
    .X(_171_));
 sky130_fd_sc_hd__clkbuf_1 _355_ (.A(_171_),
    .X(_037_));
 sky130_fd_sc_hd__mux2_1 _356_ (.A0(net55),
    .A1(net49),
    .S(_098_),
    .X(_172_));
 sky130_fd_sc_hd__clkbuf_1 _357_ (.A(_172_),
    .X(_038_));
 sky130_fd_sc_hd__mux2_1 _358_ (.A0(\u_sort.bucket_reg[2][0] ),
    .A1(net37),
    .S(_098_),
    .X(_173_));
 sky130_fd_sc_hd__clkbuf_1 _359_ (.A(_173_),
    .X(_039_));
 sky130_fd_sc_hd__mux2_1 _360_ (.A0(\u_sort.bucket_reg[2][1] ),
    .A1(net34),
    .S(_098_),
    .X(_174_));
 sky130_fd_sc_hd__clkbuf_1 _361_ (.A(net35),
    .X(_040_));
 sky130_fd_sc_hd__mux2_1 _362_ (.A0(net43),
    .A1(net36),
    .S(_098_),
    .X(_175_));
 sky130_fd_sc_hd__clkbuf_1 _363_ (.A(_175_),
    .X(_041_));
 sky130_fd_sc_hd__mux2_1 _364_ (.A0(\u_sort.bucket_reg[1][0] ),
    .A1(net42),
    .S(_098_),
    .X(_176_));
 sky130_fd_sc_hd__clkbuf_1 _365_ (.A(_176_),
    .X(_042_));
 sky130_fd_sc_hd__mux2_1 _366_ (.A0(\u_sort.bucket_reg[1][1] ),
    .A1(net53),
    .S(_098_),
    .X(_177_));
 sky130_fd_sc_hd__clkbuf_1 _367_ (.A(_177_),
    .X(_043_));
 sky130_fd_sc_hd__mux2_1 _368_ (.A0(\u_sort.bucket_reg[1][2] ),
    .A1(net38),
    .S(_098_),
    .X(_178_));
 sky130_fd_sc_hd__clkbuf_1 _369_ (.A(net39),
    .X(_044_));
 sky130_fd_sc_hd__mux2_1 _370_ (.A0(net60),
    .A1(net59),
    .S(_098_),
    .X(_179_));
 sky130_fd_sc_hd__clkbuf_1 _371_ (.A(_179_),
    .X(_045_));
 sky130_fd_sc_hd__mux2_1 _372_ (.A0(net61),
    .A1(net56),
    .S(_098_),
    .X(_180_));
 sky130_fd_sc_hd__clkbuf_1 _373_ (.A(_180_),
    .X(_046_));
 sky130_fd_sc_hd__mux2_1 _374_ (.A0(\u_sort.bucket_reg[0][2] ),
    .A1(net51),
    .S(_098_),
    .X(_181_));
 sky130_fd_sc_hd__clkbuf_1 _375_ (.A(_181_),
    .X(_047_));
 sky130_fd_sc_hd__mux2_1 _376_ (.A0(net40),
    .A1(\u_sort.bucket_counter[0] ),
    .S(_002_),
    .X(_182_));
 sky130_fd_sc_hd__clkbuf_1 _377_ (.A(_182_),
    .X(_048_));
 sky130_fd_sc_hd__mux2_1 _378_ (.A0(net46),
    .A1(_071_),
    .S(_002_),
    .X(_183_));
 sky130_fd_sc_hd__clkbuf_1 _379_ (.A(_183_),
    .X(_049_));
 sky130_fd_sc_hd__mux2_1 _380_ (.A0(net41),
    .A1(_074_),
    .S(_002_),
    .X(_184_));
 sky130_fd_sc_hd__clkbuf_1 _381_ (.A(_184_),
    .X(_050_));
 sky130_fd_sc_hd__nand2_1 _382_ (.A(net26),
    .B(_096_),
    .Y(_185_));
 sky130_fd_sc_hd__o21ai_1 _383_ (.A1(_087_),
    .A2(_185_),
    .B1(_107_),
    .Y(_186_));
 sky130_fd_sc_hd__mux2_1 _384_ (.A0(net63),
    .A1(_186_),
    .S(_002_),
    .X(_187_));
 sky130_fd_sc_hd__clkbuf_1 _385_ (.A(_187_),
    .X(_051_));
 sky130_fd_sc_hd__a21oi_1 _386_ (.A1(_092_),
    .A2(_096_),
    .B1(_109_),
    .Y(_188_));
 sky130_fd_sc_hd__mux2_1 _387_ (.A0(net66),
    .A1(_188_),
    .S(_002_),
    .X(_189_));
 sky130_fd_sc_hd__clkbuf_1 _388_ (.A(_189_),
    .X(_052_));
 sky130_fd_sc_hd__mux2_1 _389_ (.A0(net44),
    .A1(_102_),
    .S(_002_),
    .X(_190_));
 sky130_fd_sc_hd__clkbuf_1 _390_ (.A(_190_),
    .X(_053_));
 sky130_fd_sc_hd__nand2_2 _391_ (.A(\u_sort.busy ),
    .B(_097_),
    .Y(_191_));
 sky130_fd_sc_hd__or3_1 _392_ (.A(_070_),
    .B(_111_),
    .C(_191_),
    .X(_192_));
 sky130_fd_sc_hd__o21ai_1 _393_ (.A1(_111_),
    .A2(_191_),
    .B1(_070_),
    .Y(_193_));
 sky130_fd_sc_hd__and3_1 _394_ (.A(_099_),
    .B(_192_),
    .C(_193_),
    .X(_194_));
 sky130_fd_sc_hd__clkbuf_1 _395_ (.A(_194_),
    .X(_054_));
 sky130_fd_sc_hd__nor2_1 _396_ (.A(_111_),
    .B(_191_),
    .Y(_195_));
 sky130_fd_sc_hd__nand2_1 _397_ (.A(\u_sort.bucket_counter[0] ),
    .B(_071_),
    .Y(_196_));
 sky130_fd_sc_hd__or2_1 _398_ (.A(\u_sort.bucket_counter[0] ),
    .B(_071_),
    .X(_197_));
 sky130_fd_sc_hd__a211o_1 _399_ (.A1(_196_),
    .A2(_197_),
    .B1(_111_),
    .C1(_191_),
    .X(_198_));
 sky130_fd_sc_hd__o211a_1 _400_ (.A1(_071_),
    .A2(_195_),
    .B1(_198_),
    .C1(_099_),
    .X(_055_));
 sky130_fd_sc_hd__o31ai_1 _401_ (.A1(_111_),
    .A2(_191_),
    .A3(_196_),
    .B1(_074_),
    .Y(_199_));
 sky130_fd_sc_hd__or4_1 _402_ (.A(_074_),
    .B(_111_),
    .C(_191_),
    .D(_196_),
    .X(_200_));
 sky130_fd_sc_hd__a21oi_1 _403_ (.A1(_199_),
    .A2(_200_),
    .B1(_098_),
    .Y(_056_));
 sky130_fd_sc_hd__o21ai_1 _404_ (.A1(_101_),
    .A2(_106_),
    .B1(net32),
    .Y(_201_));
 sky130_fd_sc_hd__nand2_1 _405_ (.A(_099_),
    .B(_201_),
    .Y(_057_));
 sky130_fd_sc_hd__o21ai_1 _406_ (.A1(_101_),
    .A2(_107_),
    .B1(net33),
    .Y(_202_));
 sky130_fd_sc_hd__nand2_1 _407_ (.A(_099_),
    .B(_202_),
    .Y(_058_));
 sky130_fd_sc_hd__o31a_1 _408_ (.A1(_101_),
    .A2(_109_),
    .A3(_096_),
    .B1(\u_sort.valid_reg[2] ),
    .X(_203_));
 sky130_fd_sc_hd__or2_1 _409_ (.A(_098_),
    .B(_203_),
    .X(_204_));
 sky130_fd_sc_hd__clkbuf_1 _410_ (.A(_204_),
    .X(_059_));
 sky130_fd_sc_hd__o31a_1 _411_ (.A1(_101_),
    .A2(_109_),
    .A3(_185_),
    .B1(\u_sort.valid_reg[3] ),
    .X(_205_));
 sky130_fd_sc_hd__or2_1 _412_ (.A(_098_),
    .B(_205_),
    .X(_206_));
 sky130_fd_sc_hd__clkbuf_1 _413_ (.A(_206_),
    .X(_060_));
 sky130_fd_sc_hd__a21oi_1 _414_ (.A1(\u_sort.busy ),
    .A2(_102_),
    .B1(_069_),
    .Y(_207_));
 sky130_fd_sc_hd__or2_1 _415_ (.A(_098_),
    .B(_207_),
    .X(_208_));
 sky130_fd_sc_hd__clkbuf_1 _416_ (.A(_208_),
    .X(_061_));
 sky130_fd_sc_hd__xor2_1 _417_ (.A(_065_),
    .B(net62),
    .X(_062_));
 sky130_fd_sc_hd__a21oi_1 _418_ (.A1(_065_),
    .A2(\u_am0.i_index[0] ),
    .B1(\u_am0.i_index[1] ),
    .Y(_209_));
 sky130_fd_sc_hd__nor2_1 _419_ (.A(_066_),
    .B(_209_),
    .Y(_063_));
 sky130_fd_sc_hd__nor2_1 _420_ (.A(\u_am0.i_index[2] ),
    .B(_066_),
    .Y(_210_));
 sky130_fd_sc_hd__nor2_1 _421_ (.A(_000_),
    .B(_210_),
    .Y(_064_));
 sky130_fd_sc_hd__dfrtp_1 _422_ (.CLK(clknet_3_2__leaf_clk),
    .D(_000_),
    .RESET_B(net17),
    .Q(\u_control.o_start_sort ));
 sky130_fd_sc_hd__dfrtp_1 _423_ (.CLK(clknet_3_3__leaf_clk),
    .D(_003_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _424_ (.CLK(clknet_3_3__leaf_clk),
    .D(_004_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _425_ (.CLK(clknet_3_3__leaf_clk),
    .D(_005_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _426_ (.CLK(clknet_3_7__leaf_clk),
    .D(_006_),
    .RESET_B(net17),
    .Q(\u_am0.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _427_ (.CLK(clknet_3_7__leaf_clk),
    .D(_007_),
    .RESET_B(net17),
    .Q(\u_am0.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _428_ (.CLK(clknet_3_7__leaf_clk),
    .D(_008_),
    .RESET_B(net17),
    .Q(\u_am0.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _429_ (.CLK(clknet_3_5__leaf_clk),
    .D(_009_),
    .RESET_B(net17),
    .Q(\u_am1.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _430_ (.CLK(clknet_3_5__leaf_clk),
    .D(_010_),
    .RESET_B(net17),
    .Q(\u_am1.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _431_ (.CLK(clknet_3_5__leaf_clk),
    .D(_011_),
    .RESET_B(net17),
    .Q(\u_am1.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _432_ (.CLK(clknet_3_7__leaf_clk),
    .D(_012_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _433_ (.CLK(clknet_3_6__leaf_clk),
    .D(_013_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _434_ (.CLK(clknet_3_6__leaf_clk),
    .D(_014_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _435_ (.CLK(clknet_3_1__leaf_clk),
    .D(_015_),
    .RESET_B(net17),
    .Q(\u_am2.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _436_ (.CLK(clknet_3_0__leaf_clk),
    .D(_016_),
    .RESET_B(net17),
    .Q(\u_am2.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _437_ (.CLK(clknet_3_0__leaf_clk),
    .D(_017_),
    .RESET_B(net17),
    .Q(\u_am2.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _438_ (.CLK(clknet_3_5__leaf_clk),
    .D(_018_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _439_ (.CLK(clknet_3_4__leaf_clk),
    .D(_019_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _440_ (.CLK(clknet_3_4__leaf_clk),
    .D(_020_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _441_ (.CLK(clknet_3_0__leaf_clk),
    .D(_021_),
    .RESET_B(net17),
    .Q(\u_am3.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _442_ (.CLK(clknet_3_0__leaf_clk),
    .D(_022_),
    .RESET_B(net17),
    .Q(\u_am3.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _443_ (.CLK(clknet_3_0__leaf_clk),
    .D(_023_),
    .RESET_B(net17),
    .Q(\u_am3.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _444_ (.CLK(clknet_3_1__leaf_clk),
    .D(_024_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _445_ (.CLK(clknet_3_1__leaf_clk),
    .D(_025_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _446_ (.CLK(clknet_3_1__leaf_clk),
    .D(_026_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _447_ (.CLK(clknet_3_2__leaf_clk),
    .D(_027_),
    .RESET_B(net17),
    .Q(\u_am4.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _448_ (.CLK(clknet_3_2__leaf_clk),
    .D(_028_),
    .RESET_B(net17),
    .Q(\u_am4.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _449_ (.CLK(clknet_3_2__leaf_clk),
    .D(_029_),
    .RESET_B(net17),
    .Q(\u_am4.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _450_ (.CLK(clknet_3_0__leaf_clk),
    .D(_030_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _451_ (.CLK(clknet_3_0__leaf_clk),
    .D(_031_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _452_ (.CLK(clknet_3_0__leaf_clk),
    .D(_032_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _453_ (.CLK(clknet_3_3__leaf_clk),
    .D(_033_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][0] ));
 sky130_fd_sc_hd__dfrtp_1 _454_ (.CLK(clknet_3_3__leaf_clk),
    .D(_034_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][1] ));
 sky130_fd_sc_hd__dfrtp_1 _455_ (.CLK(clknet_3_3__leaf_clk),
    .D(_035_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][2] ));
 sky130_fd_sc_hd__dfrtp_1 _456_ (.CLK(clknet_3_2__leaf_clk),
    .D(_036_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][0] ));
 sky130_fd_sc_hd__dfrtp_1 _457_ (.CLK(clknet_3_0__leaf_clk),
    .D(_037_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][1] ));
 sky130_fd_sc_hd__dfrtp_1 _458_ (.CLK(clknet_3_0__leaf_clk),
    .D(_038_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][2] ));
 sky130_fd_sc_hd__dfrtp_1 _459_ (.CLK(clknet_3_1__leaf_clk),
    .D(_039_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][0] ));
 sky130_fd_sc_hd__dfrtp_1 _460_ (.CLK(clknet_3_0__leaf_clk),
    .D(_040_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][1] ));
 sky130_fd_sc_hd__dfrtp_1 _461_ (.CLK(clknet_3_1__leaf_clk),
    .D(_041_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][2] ));
 sky130_fd_sc_hd__dfrtp_1 _462_ (.CLK(clknet_3_5__leaf_clk),
    .D(_042_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][0] ));
 sky130_fd_sc_hd__dfrtp_1 _463_ (.CLK(clknet_3_4__leaf_clk),
    .D(_043_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][1] ));
 sky130_fd_sc_hd__dfrtp_1 _464_ (.CLK(clknet_3_4__leaf_clk),
    .D(_044_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][2] ));
 sky130_fd_sc_hd__dfrtp_1 _465_ (.CLK(clknet_3_7__leaf_clk),
    .D(_045_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][0] ));
 sky130_fd_sc_hd__dfrtp_1 _466_ (.CLK(clknet_3_6__leaf_clk),
    .D(_046_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][1] ));
 sky130_fd_sc_hd__dfrtp_1 _467_ (.CLK(clknet_3_6__leaf_clk),
    .D(_047_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][2] ));
 sky130_fd_sc_hd__dfrtp_1 _468_ (.CLK(clknet_3_6__leaf_clk),
    .D(_048_),
    .RESET_B(net17),
    .Q(net18));
 sky130_fd_sc_hd__dfrtp_1 _469_ (.CLK(clknet_3_7__leaf_clk),
    .D(_049_),
    .RESET_B(net17),
    .Q(net19));
 sky130_fd_sc_hd__dfrtp_1 _470_ (.CLK(clknet_3_5__leaf_clk),
    .D(_050_),
    .RESET_B(net17),
    .Q(net20));
 sky130_fd_sc_hd__dfrtp_1 _471_ (.CLK(clknet_3_5__leaf_clk),
    .D(_002_),
    .RESET_B(net17),
    .Q(net24));
 sky130_fd_sc_hd__dfrtp_1 _472_ (.CLK(clknet_3_7__leaf_clk),
    .D(_051_),
    .RESET_B(net17),
    .Q(net21));
 sky130_fd_sc_hd__dfrtp_1 _473_ (.CLK(clknet_3_3__leaf_clk),
    .D(_052_),
    .RESET_B(net17),
    .Q(net22));
 sky130_fd_sc_hd__dfrtp_1 _474_ (.CLK(clknet_3_4__leaf_clk),
    .D(_053_),
    .RESET_B(net17),
    .Q(net23));
 sky130_fd_sc_hd__dfrtp_4 _475_ (.CLK(clknet_3_7__leaf_clk),
    .D(_054_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[0] ));
 sky130_fd_sc_hd__dfrtp_1 _476_ (.CLK(clknet_3_7__leaf_clk),
    .D(_055_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[1] ));
 sky130_fd_sc_hd__dfrtp_1 _477_ (.CLK(clknet_3_7__leaf_clk),
    .D(_056_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[2] ));
 sky130_fd_sc_hd__dfrtp_1 _478_ (.CLK(clknet_3_6__leaf_clk),
    .D(_001_),
    .RESET_B(net17),
    .Q(\u_sort.busy ));
 sky130_fd_sc_hd__dfrtp_1 _479_ (.CLK(clknet_3_6__leaf_clk),
    .D(_057_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[0] ));
 sky130_fd_sc_hd__dfrtp_1 _480_ (.CLK(clknet_3_7__leaf_clk),
    .D(_058_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[1] ));
 sky130_fd_sc_hd__dfrtp_1 _481_ (.CLK(clknet_3_4__leaf_clk),
    .D(_059_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[2] ));
 sky130_fd_sc_hd__dfrtp_1 _482_ (.CLK(clknet_3_3__leaf_clk),
    .D(_060_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[3] ));
 sky130_fd_sc_hd__dfrtp_1 _483_ (.CLK(clknet_3_6__leaf_clk),
    .D(_061_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[4] ));
 sky130_fd_sc_hd__dfrtp_4 _484_ (.CLK(clknet_3_2__leaf_clk),
    .D(_062_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[0] ));
 sky130_fd_sc_hd__dfrtp_4 _485_ (.CLK(clknet_3_2__leaf_clk),
    .D(_063_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[1] ));
 sky130_fd_sc_hd__dfrtp_4 _486_ (.CLK(clknet_3_2__leaf_clk),
    .D(_064_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[2] ));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_0_clk (.A(clk),
    .X(clknet_0_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_0__f_clk (.A(clknet_0_clk),
    .X(clknet_3_0__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_1__f_clk (.A(clknet_0_clk),
    .X(clknet_3_1__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_2__f_clk (.A(clknet_0_clk),
    .X(clknet_3_2__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_3__f_clk (.A(clknet_0_clk),
    .X(clknet_3_3__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_4__f_clk (.A(clknet_0_clk),
    .X(clknet_3_4__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_5__f_clk (.A(clknet_0_clk),
    .X(clknet_3_5__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_6__f_clk (.A(clknet_0_clk),
    .X(clknet_3_6__leaf_clk));
 sky130_fd_sc_hd__clkbuf_16 clkbuf_3_7__f_clk (.A(clknet_0_clk),
    .X(clknet_3_7__leaf_clk));
 sky130_fd_sc_hd__dlygate4sd3_1 hold1 (.A(\u_am1.max[2] ),
    .X(net27));
 sky130_fd_sc_hd__dlygate4sd3_1 hold10 (.A(\u_am2.b_i[2] ),
    .X(net36));
 sky130_fd_sc_hd__dlygate4sd3_1 hold11 (.A(\u_am2.b_i[0] ),
    .X(net37));
 sky130_fd_sc_hd__dlygate4sd3_1 hold12 (.A(\u_am1.b_i[2] ),
    .X(net38));
 sky130_fd_sc_hd__dlygate4sd3_1 hold13 (.A(_178_),
    .X(net39));
 sky130_fd_sc_hd__dlygate4sd3_1 hold14 (.A(net18),
    .X(net40));
 sky130_fd_sc_hd__dlygate4sd3_1 hold15 (.A(net20),
    .X(net41));
 sky130_fd_sc_hd__dlygate4sd3_1 hold16 (.A(\u_am1.b_i[0] ),
    .X(net42));
 sky130_fd_sc_hd__dlygate4sd3_1 hold17 (.A(\u_sort.bucket_reg[2][2] ),
    .X(net43));
 sky130_fd_sc_hd__dlygate4sd3_1 hold18 (.A(net23),
    .X(net44));
 sky130_fd_sc_hd__dlygate4sd3_1 hold19 (.A(\u_am3.b_i[1] ),
    .X(net45));
 sky130_fd_sc_hd__dlygate4sd3_1 hold2 (.A(\u_am4.max[2] ),
    .X(net28));
 sky130_fd_sc_hd__dlygate4sd3_1 hold20 (.A(net19),
    .X(net46));
 sky130_fd_sc_hd__dlygate4sd3_1 hold21 (.A(\u_am4.b_i[1] ),
    .X(net47));
 sky130_fd_sc_hd__dlygate4sd3_1 hold22 (.A(\u_sort.bucket_reg[4][1] ),
    .X(net48));
 sky130_fd_sc_hd__dlygate4sd3_1 hold23 (.A(\u_am3.b_i[2] ),
    .X(net49));
 sky130_fd_sc_hd__dlygate4sd3_1 hold24 (.A(\u_am4.b_i[2] ),
    .X(net50));
 sky130_fd_sc_hd__dlygate4sd3_1 hold25 (.A(\u_am0.b_i[2] ),
    .X(net51));
 sky130_fd_sc_hd__dlygate4sd3_1 hold26 (.A(\u_sort.bucket_reg[3][1] ),
    .X(net52));
 sky130_fd_sc_hd__dlygate4sd3_1 hold27 (.A(\u_am1.b_i[1] ),
    .X(net53));
 sky130_fd_sc_hd__dlygate4sd3_1 hold28 (.A(\u_sort.bucket_reg[4][2] ),
    .X(net54));
 sky130_fd_sc_hd__dlygate4sd3_1 hold29 (.A(\u_sort.bucket_reg[3][2] ),
    .X(net55));
 sky130_fd_sc_hd__dlygate4sd3_1 hold3 (.A(\u_am3.max[2] ),
    .X(net29));
 sky130_fd_sc_hd__dlygate4sd3_1 hold30 (.A(\u_am0.b_i[1] ),
    .X(net56));
 sky130_fd_sc_hd__dlygate4sd3_1 hold31 (.A(\u_am4.b_i[0] ),
    .X(net57));
 sky130_fd_sc_hd__dlygate4sd3_1 hold32 (.A(\u_am3.b_i[0] ),
    .X(net58));
 sky130_fd_sc_hd__dlygate4sd3_1 hold33 (.A(\u_am0.b_i[0] ),
    .X(net59));
 sky130_fd_sc_hd__dlygate4sd3_1 hold34 (.A(\u_sort.bucket_reg[0][0] ),
    .X(net60));
 sky130_fd_sc_hd__dlygate4sd3_1 hold35 (.A(\u_sort.bucket_reg[0][1] ),
    .X(net61));
 sky130_fd_sc_hd__dlygate4sd3_1 hold36 (.A(\u_am0.i_index[0] ),
    .X(net62));
 sky130_fd_sc_hd__dlygate4sd3_1 hold37 (.A(net21),
    .X(net63));
 sky130_fd_sc_hd__dlygate4sd3_1 hold38 (.A(\u_am2.max[0] ),
    .X(net64));
 sky130_fd_sc_hd__dlygate4sd3_1 hold39 (.A(\u_am4.max[0] ),
    .X(net65));
 sky130_fd_sc_hd__dlygate4sd3_1 hold4 (.A(\u_am0.max[2] ),
    .X(net30));
 sky130_fd_sc_hd__dlygate4sd3_1 hold40 (.A(net22),
    .X(net66));
 sky130_fd_sc_hd__dlygate4sd3_1 hold41 (.A(\u_am1.max[0] ),
    .X(net67));
 sky130_fd_sc_hd__dlygate4sd3_1 hold42 (.A(\u_am0.max[0] ),
    .X(net68));
 sky130_fd_sc_hd__dlygate4sd3_1 hold43 (.A(\u_am3.max[0] ),
    .X(net69));
 sky130_fd_sc_hd__dlygate4sd3_1 hold44 (.A(\u_am1.max[1] ),
    .X(net70));
 sky130_fd_sc_hd__dlygate4sd3_1 hold45 (.A(\u_am2.max[1] ),
    .X(net71));
 sky130_fd_sc_hd__dlygate4sd3_1 hold46 (.A(\u_am4.max[1] ),
    .X(net72));
 sky130_fd_sc_hd__dlygate4sd3_1 hold47 (.A(\u_am0.max[1] ),
    .X(net73));
 sky130_fd_sc_hd__dlygate4sd3_1 hold5 (.A(\u_am2.max[2] ),
    .X(net31));
 sky130_fd_sc_hd__dlygate4sd3_1 hold6 (.A(\u_sort.valid_reg[0] ),
    .X(net32));
 sky130_fd_sc_hd__dlygate4sd3_1 hold7 (.A(\u_sort.valid_reg[1] ),
    .X(net33));
 sky130_fd_sc_hd__dlygate4sd3_1 hold8 (.A(\u_am2.b_i[1] ),
    .X(net34));
 sky130_fd_sc_hd__dlygate4sd3_1 hold9 (.A(_174_),
    .X(net35));
 sky130_fd_sc_hd__clkbuf_1 input1 (.A(i_valid),
    .X(net1));
 sky130_fd_sc_hd__buf_1 input10 (.A(i_value2[2]),
    .X(net10));
 sky130_fd_sc_hd__buf_1 input11 (.A(i_value3[0]),
    .X(net11));
 sky130_fd_sc_hd__buf_1 input12 (.A(i_value3[1]),
    .X(net12));
 sky130_fd_sc_hd__buf_1 input13 (.A(i_value3[2]),
    .X(net13));
 sky130_fd_sc_hd__buf_1 input14 (.A(i_value4[0]),
    .X(net14));
 sky130_fd_sc_hd__buf_1 input15 (.A(i_value4[1]),
    .X(net15));
 sky130_fd_sc_hd__buf_1 input16 (.A(i_value4[2]),
    .X(net16));
 sky130_fd_sc_hd__buf_12 input17 (.A(rst_n),
    .X(net17));
 sky130_fd_sc_hd__buf_1 input2 (.A(i_value0[0]),
    .X(net2));
 sky130_fd_sc_hd__buf_1 input3 (.A(i_value0[1]),
    .X(net3));
 sky130_fd_sc_hd__buf_1 input4 (.A(i_value0[2]),
    .X(net4));
 sky130_fd_sc_hd__clkbuf_1 input5 (.A(i_value1[0]),
    .X(net5));
 sky130_fd_sc_hd__buf_1 input6 (.A(i_value1[1]),
    .X(net6));
 sky130_fd_sc_hd__buf_1 input7 (.A(i_value1[2]),
    .X(net7));
 sky130_fd_sc_hd__buf_1 input8 (.A(i_value2[0]),
    .X(net8));
 sky130_fd_sc_hd__buf_1 input9 (.A(i_value2[1]),
    .X(net9));
 sky130_fd_sc_hd__clkbuf_1 max_cap25 (.A(net26),
    .X(net25));
 sky130_fd_sc_hd__buf_2 output18 (.A(net18),
    .X(o_bucket[0]));
 sky130_fd_sc_hd__buf_2 output19 (.A(net19),
    .X(o_bucket[1]));
 sky130_fd_sc_hd__clkbuf_4 output20 (.A(net20),
    .X(o_bucket[2]));
 sky130_fd_sc_hd__buf_2 output21 (.A(net21),
    .X(o_index[0]));
 sky130_fd_sc_hd__clkbuf_4 output22 (.A(net22),
    .X(o_index[1]));
 sky130_fd_sc_hd__clkbuf_4 output23 (.A(net23),
    .X(o_index[2]));
 sky130_fd_sc_hd__clkbuf_4 output24 (.A(net24),
    .X(o_valid));
 sky130_fd_sc_hd__clkbuf_1 wire26 (.A(_103_),
    .X(net26));
endmodule

