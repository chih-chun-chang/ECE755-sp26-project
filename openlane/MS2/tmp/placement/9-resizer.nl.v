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
 wire net1;
 wire net2;
 wire net3;
 wire net4;
 wire net5;
 wire net6;
 wire net7;
 wire net8;
 wire net9;
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
 wire net20;
 wire net21;
 wire net22;
 wire net23;
 wire net24;
 wire net25;
 wire net26;

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
 sky130_fd_sc_hd__mux2_1 _267_ (.A0(\u_am4.b_i[0] ),
    .A1(\u_am0.i_index[0] ),
    .S(_117_),
    .X(_118_));
 sky130_fd_sc_hd__clkbuf_1 _268_ (.A(_118_),
    .X(_003_));
 sky130_fd_sc_hd__mux2_1 _269_ (.A0(\u_am4.b_i[1] ),
    .A1(\u_am0.i_index[1] ),
    .S(_117_),
    .X(_119_));
 sky130_fd_sc_hd__clkbuf_1 _270_ (.A(_119_),
    .X(_004_));
 sky130_fd_sc_hd__mux2_1 _271_ (.A0(\u_am4.b_i[2] ),
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
 sky130_fd_sc_hd__mux2_1 _279_ (.A0(\u_am0.max[0] ),
    .A1(net2),
    .S(_126_),
    .X(_127_));
 sky130_fd_sc_hd__clkbuf_1 _280_ (.A(_127_),
    .X(_006_));
 sky130_fd_sc_hd__mux2_1 _281_ (.A0(\u_am0.max[1] ),
    .A1(net3),
    .S(_126_),
    .X(_128_));
 sky130_fd_sc_hd__clkbuf_1 _282_ (.A(_128_),
    .X(_007_));
 sky130_fd_sc_hd__a21o_1 _283_ (.A1(_065_),
    .A2(net4),
    .B1(\u_am0.max[2] ),
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
 sky130_fd_sc_hd__mux2_1 _290_ (.A0(\u_am1.max[0] ),
    .A1(net5),
    .S(_134_),
    .X(_135_));
 sky130_fd_sc_hd__clkbuf_1 _291_ (.A(_135_),
    .X(_009_));
 sky130_fd_sc_hd__mux2_1 _292_ (.A0(\u_am1.max[1] ),
    .A1(net6),
    .S(_134_),
    .X(_136_));
 sky130_fd_sc_hd__clkbuf_1 _293_ (.A(_136_),
    .X(_010_));
 sky130_fd_sc_hd__a21o_1 _294_ (.A1(_065_),
    .A2(net7),
    .B1(\u_am1.max[2] ),
    .X(_011_));
 sky130_fd_sc_hd__mux2_1 _295_ (.A0(\u_am0.b_i[0] ),
    .A1(\u_am0.i_index[0] ),
    .S(_126_),
    .X(_137_));
 sky130_fd_sc_hd__clkbuf_1 _296_ (.A(_137_),
    .X(_012_));
 sky130_fd_sc_hd__mux2_1 _297_ (.A0(\u_am0.b_i[1] ),
    .A1(\u_am0.i_index[1] ),
    .S(_126_),
    .X(_138_));
 sky130_fd_sc_hd__clkbuf_1 _298_ (.A(_138_),
    .X(_013_));
 sky130_fd_sc_hd__mux2_1 _299_ (.A0(\u_am0.b_i[2] ),
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
 sky130_fd_sc_hd__mux2_1 _307_ (.A0(\u_am2.max[0] ),
    .A1(net8),
    .S(_145_),
    .X(_146_));
 sky130_fd_sc_hd__clkbuf_1 _308_ (.A(_146_),
    .X(_015_));
 sky130_fd_sc_hd__mux2_1 _309_ (.A0(\u_am2.max[1] ),
    .A1(net9),
    .S(_145_),
    .X(_147_));
 sky130_fd_sc_hd__clkbuf_1 _310_ (.A(_147_),
    .X(_016_));
 sky130_fd_sc_hd__a21o_1 _311_ (.A1(_065_),
    .A2(net10),
    .B1(\u_am2.max[2] ),
    .X(_017_));
 sky130_fd_sc_hd__mux2_1 _312_ (.A0(\u_am1.b_i[0] ),
    .A1(\u_am0.i_index[0] ),
    .S(_134_),
    .X(_148_));
 sky130_fd_sc_hd__clkbuf_1 _313_ (.A(_148_),
    .X(_018_));
 sky130_fd_sc_hd__mux2_1 _314_ (.A0(\u_am1.b_i[1] ),
    .A1(\u_am0.i_index[1] ),
    .S(_134_),
    .X(_149_));
 sky130_fd_sc_hd__clkbuf_1 _315_ (.A(_149_),
    .X(_019_));
 sky130_fd_sc_hd__mux2_1 _316_ (.A0(\u_am1.b_i[2] ),
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
 sky130_fd_sc_hd__mux2_1 _324_ (.A0(\u_am3.max[0] ),
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
    .B1(\u_am3.max[2] ),
    .X(_023_));
 sky130_fd_sc_hd__mux2_1 _329_ (.A0(\u_am2.b_i[0] ),
    .A1(\u_am0.i_index[0] ),
    .S(_145_),
    .X(_159_));
 sky130_fd_sc_hd__clkbuf_1 _330_ (.A(_159_),
    .X(_024_));
 sky130_fd_sc_hd__mux2_1 _331_ (.A0(\u_am2.b_i[1] ),
    .A1(\u_am0.i_index[1] ),
    .S(_145_),
    .X(_160_));
 sky130_fd_sc_hd__clkbuf_1 _332_ (.A(_160_),
    .X(_025_));
 sky130_fd_sc_hd__mux2_1 _333_ (.A0(\u_am2.b_i[2] ),
    .A1(\u_am0.i_index[2] ),
    .S(_145_),
    .X(_161_));
 sky130_fd_sc_hd__clkbuf_1 _334_ (.A(_161_),
    .X(_026_));
 sky130_fd_sc_hd__mux2_1 _335_ (.A0(\u_am4.max[0] ),
    .A1(net14),
    .S(_117_),
    .X(_162_));
 sky130_fd_sc_hd__clkbuf_1 _336_ (.A(_162_),
    .X(_027_));
 sky130_fd_sc_hd__mux2_1 _337_ (.A0(\u_am4.max[1] ),
    .A1(net15),
    .S(_117_),
    .X(_163_));
 sky130_fd_sc_hd__clkbuf_1 _338_ (.A(_163_),
    .X(_028_));
 sky130_fd_sc_hd__a21o_1 _339_ (.A1(net16),
    .A2(_065_),
    .B1(\u_am4.max[2] ),
    .X(_029_));
 sky130_fd_sc_hd__mux2_1 _340_ (.A0(\u_am3.b_i[0] ),
    .A1(\u_am0.i_index[0] ),
    .S(_156_),
    .X(_164_));
 sky130_fd_sc_hd__clkbuf_1 _341_ (.A(_164_),
    .X(_030_));
 sky130_fd_sc_hd__mux2_1 _342_ (.A0(\u_am3.b_i[1] ),
    .A1(\u_am0.i_index[1] ),
    .S(_156_),
    .X(_165_));
 sky130_fd_sc_hd__clkbuf_1 _343_ (.A(_165_),
    .X(_031_));
 sky130_fd_sc_hd__mux2_1 _344_ (.A0(\u_am3.b_i[2] ),
    .A1(\u_am0.i_index[2] ),
    .S(_156_),
    .X(_166_));
 sky130_fd_sc_hd__clkbuf_1 _345_ (.A(_166_),
    .X(_032_));
 sky130_fd_sc_hd__mux2_1 _346_ (.A0(\u_sort.bucket_reg[4][0] ),
    .A1(\u_am4.b_i[0] ),
    .S(_098_),
    .X(_167_));
 sky130_fd_sc_hd__clkbuf_1 _347_ (.A(_167_),
    .X(_033_));
 sky130_fd_sc_hd__mux2_1 _348_ (.A0(\u_sort.bucket_reg[4][1] ),
    .A1(\u_am4.b_i[1] ),
    .S(_098_),
    .X(_168_));
 sky130_fd_sc_hd__clkbuf_1 _349_ (.A(_168_),
    .X(_034_));
 sky130_fd_sc_hd__mux2_1 _350_ (.A0(\u_sort.bucket_reg[4][2] ),
    .A1(\u_am4.b_i[2] ),
    .S(_098_),
    .X(_169_));
 sky130_fd_sc_hd__clkbuf_1 _351_ (.A(_169_),
    .X(_035_));
 sky130_fd_sc_hd__mux2_1 _352_ (.A0(\u_sort.bucket_reg[3][0] ),
    .A1(\u_am3.b_i[0] ),
    .S(_098_),
    .X(_170_));
 sky130_fd_sc_hd__clkbuf_1 _353_ (.A(_170_),
    .X(_036_));
 sky130_fd_sc_hd__mux2_1 _354_ (.A0(\u_sort.bucket_reg[3][1] ),
    .A1(\u_am3.b_i[1] ),
    .S(_098_),
    .X(_171_));
 sky130_fd_sc_hd__clkbuf_1 _355_ (.A(_171_),
    .X(_037_));
 sky130_fd_sc_hd__mux2_1 _356_ (.A0(\u_sort.bucket_reg[3][2] ),
    .A1(\u_am3.b_i[2] ),
    .S(_098_),
    .X(_172_));
 sky130_fd_sc_hd__clkbuf_1 _357_ (.A(_172_),
    .X(_038_));
 sky130_fd_sc_hd__mux2_1 _358_ (.A0(\u_sort.bucket_reg[2][0] ),
    .A1(\u_am2.b_i[0] ),
    .S(_098_),
    .X(_173_));
 sky130_fd_sc_hd__clkbuf_1 _359_ (.A(_173_),
    .X(_039_));
 sky130_fd_sc_hd__mux2_1 _360_ (.A0(\u_sort.bucket_reg[2][1] ),
    .A1(\u_am2.b_i[1] ),
    .S(_098_),
    .X(_174_));
 sky130_fd_sc_hd__clkbuf_1 _361_ (.A(_174_),
    .X(_040_));
 sky130_fd_sc_hd__mux2_1 _362_ (.A0(\u_sort.bucket_reg[2][2] ),
    .A1(\u_am2.b_i[2] ),
    .S(_098_),
    .X(_175_));
 sky130_fd_sc_hd__clkbuf_1 _363_ (.A(_175_),
    .X(_041_));
 sky130_fd_sc_hd__mux2_1 _364_ (.A0(\u_sort.bucket_reg[1][0] ),
    .A1(\u_am1.b_i[0] ),
    .S(_098_),
    .X(_176_));
 sky130_fd_sc_hd__clkbuf_1 _365_ (.A(_176_),
    .X(_042_));
 sky130_fd_sc_hd__mux2_1 _366_ (.A0(\u_sort.bucket_reg[1][1] ),
    .A1(\u_am1.b_i[1] ),
    .S(_098_),
    .X(_177_));
 sky130_fd_sc_hd__clkbuf_1 _367_ (.A(_177_),
    .X(_043_));
 sky130_fd_sc_hd__mux2_1 _368_ (.A0(\u_sort.bucket_reg[1][2] ),
    .A1(\u_am1.b_i[2] ),
    .S(_098_),
    .X(_178_));
 sky130_fd_sc_hd__clkbuf_1 _369_ (.A(_178_),
    .X(_044_));
 sky130_fd_sc_hd__mux2_1 _370_ (.A0(\u_sort.bucket_reg[0][0] ),
    .A1(\u_am0.b_i[0] ),
    .S(_098_),
    .X(_179_));
 sky130_fd_sc_hd__clkbuf_1 _371_ (.A(_179_),
    .X(_045_));
 sky130_fd_sc_hd__mux2_1 _372_ (.A0(\u_sort.bucket_reg[0][1] ),
    .A1(\u_am0.b_i[1] ),
    .S(_098_),
    .X(_180_));
 sky130_fd_sc_hd__clkbuf_1 _373_ (.A(_180_),
    .X(_046_));
 sky130_fd_sc_hd__mux2_1 _374_ (.A0(\u_sort.bucket_reg[0][2] ),
    .A1(\u_am0.b_i[2] ),
    .S(_098_),
    .X(_181_));
 sky130_fd_sc_hd__clkbuf_1 _375_ (.A(_181_),
    .X(_047_));
 sky130_fd_sc_hd__mux2_1 _376_ (.A0(net18),
    .A1(\u_sort.bucket_counter[0] ),
    .S(_002_),
    .X(_182_));
 sky130_fd_sc_hd__clkbuf_1 _377_ (.A(_182_),
    .X(_048_));
 sky130_fd_sc_hd__mux2_1 _378_ (.A0(net19),
    .A1(_071_),
    .S(_002_),
    .X(_183_));
 sky130_fd_sc_hd__clkbuf_1 _379_ (.A(_183_),
    .X(_049_));
 sky130_fd_sc_hd__mux2_1 _380_ (.A0(net20),
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
 sky130_fd_sc_hd__mux2_1 _384_ (.A0(net21),
    .A1(_186_),
    .S(_002_),
    .X(_187_));
 sky130_fd_sc_hd__clkbuf_1 _385_ (.A(_187_),
    .X(_051_));
 sky130_fd_sc_hd__a21oi_1 _386_ (.A1(_092_),
    .A2(_096_),
    .B1(_109_),
    .Y(_188_));
 sky130_fd_sc_hd__mux2_1 _387_ (.A0(net22),
    .A1(_188_),
    .S(_002_),
    .X(_189_));
 sky130_fd_sc_hd__clkbuf_1 _388_ (.A(_189_),
    .X(_052_));
 sky130_fd_sc_hd__mux2_1 _389_ (.A0(net23),
    .A1(_102_),
    .S(_002_),
    .X(_190_));
 sky130_fd_sc_hd__clkbuf_1 _390_ (.A(_190_),
    .X(_053_));
 sky130_fd_sc_hd__nand2_1 _391_ (.A(\u_sort.busy ),
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
    .B1(\u_sort.valid_reg[0] ),
    .Y(_201_));
 sky130_fd_sc_hd__nand2_1 _405_ (.A(_099_),
    .B(_201_),
    .Y(_057_));
 sky130_fd_sc_hd__o21ai_1 _406_ (.A1(_101_),
    .A2(_107_),
    .B1(\u_sort.valid_reg[1] ),
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
    .B(\u_am0.i_index[0] ),
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
 sky130_fd_sc_hd__dfrtp_1 _422_ (.CLK(clk),
    .D(_000_),
    .RESET_B(net17),
    .Q(\u_control.o_start_sort ));
 sky130_fd_sc_hd__dfrtp_1 _423_ (.CLK(clk),
    .D(_003_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _424_ (.CLK(clk),
    .D(_004_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _425_ (.CLK(clk),
    .D(_005_),
    .RESET_B(net17),
    .Q(\u_am4.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _426_ (.CLK(clk),
    .D(_006_),
    .RESET_B(net17),
    .Q(\u_am0.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _427_ (.CLK(clk),
    .D(_007_),
    .RESET_B(net17),
    .Q(\u_am0.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _428_ (.CLK(clk),
    .D(_008_),
    .RESET_B(net17),
    .Q(\u_am0.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _429_ (.CLK(clk),
    .D(_009_),
    .RESET_B(net17),
    .Q(\u_am1.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _430_ (.CLK(clk),
    .D(_010_),
    .RESET_B(net17),
    .Q(\u_am1.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _431_ (.CLK(clk),
    .D(_011_),
    .RESET_B(net17),
    .Q(\u_am1.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _432_ (.CLK(clk),
    .D(_012_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _433_ (.CLK(clk),
    .D(_013_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _434_ (.CLK(clk),
    .D(_014_),
    .RESET_B(net17),
    .Q(\u_am0.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _435_ (.CLK(clk),
    .D(_015_),
    .RESET_B(net17),
    .Q(\u_am2.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _436_ (.CLK(clk),
    .D(_016_),
    .RESET_B(net17),
    .Q(\u_am2.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _437_ (.CLK(clk),
    .D(_017_),
    .RESET_B(net17),
    .Q(\u_am2.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _438_ (.CLK(clk),
    .D(_018_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _439_ (.CLK(clk),
    .D(_019_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _440_ (.CLK(clk),
    .D(_020_),
    .RESET_B(net17),
    .Q(\u_am1.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _441_ (.CLK(clk),
    .D(_021_),
    .RESET_B(net17),
    .Q(\u_am3.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _442_ (.CLK(clk),
    .D(_022_),
    .RESET_B(net17),
    .Q(\u_am3.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _443_ (.CLK(clk),
    .D(_023_),
    .RESET_B(net17),
    .Q(\u_am3.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _444_ (.CLK(clk),
    .D(_024_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _445_ (.CLK(clk),
    .D(_025_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _446_ (.CLK(clk),
    .D(_026_),
    .RESET_B(net17),
    .Q(\u_am2.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _447_ (.CLK(clk),
    .D(_027_),
    .RESET_B(net17),
    .Q(\u_am4.max[0] ));
 sky130_fd_sc_hd__dfrtp_1 _448_ (.CLK(clk),
    .D(_028_),
    .RESET_B(net17),
    .Q(\u_am4.max[1] ));
 sky130_fd_sc_hd__dfrtp_1 _449_ (.CLK(clk),
    .D(_029_),
    .RESET_B(net17),
    .Q(\u_am4.max[2] ));
 sky130_fd_sc_hd__dfrtp_1 _450_ (.CLK(clk),
    .D(_030_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[0] ));
 sky130_fd_sc_hd__dfrtp_1 _451_ (.CLK(clk),
    .D(_031_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[1] ));
 sky130_fd_sc_hd__dfrtp_1 _452_ (.CLK(clk),
    .D(_032_),
    .RESET_B(net17),
    .Q(\u_am3.b_i[2] ));
 sky130_fd_sc_hd__dfrtp_1 _453_ (.CLK(clk),
    .D(_033_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][0] ));
 sky130_fd_sc_hd__dfrtp_1 _454_ (.CLK(clk),
    .D(_034_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][1] ));
 sky130_fd_sc_hd__dfrtp_1 _455_ (.CLK(clk),
    .D(_035_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[4][2] ));
 sky130_fd_sc_hd__dfrtp_1 _456_ (.CLK(clk),
    .D(_036_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][0] ));
 sky130_fd_sc_hd__dfrtp_1 _457_ (.CLK(clk),
    .D(_037_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][1] ));
 sky130_fd_sc_hd__dfrtp_1 _458_ (.CLK(clk),
    .D(_038_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[3][2] ));
 sky130_fd_sc_hd__dfrtp_1 _459_ (.CLK(clk),
    .D(_039_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][0] ));
 sky130_fd_sc_hd__dfrtp_1 _460_ (.CLK(clk),
    .D(_040_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][1] ));
 sky130_fd_sc_hd__dfrtp_1 _461_ (.CLK(clk),
    .D(_041_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[2][2] ));
 sky130_fd_sc_hd__dfrtp_1 _462_ (.CLK(clk),
    .D(_042_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][0] ));
 sky130_fd_sc_hd__dfrtp_1 _463_ (.CLK(clk),
    .D(_043_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][1] ));
 sky130_fd_sc_hd__dfrtp_1 _464_ (.CLK(clk),
    .D(_044_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[1][2] ));
 sky130_fd_sc_hd__dfrtp_1 _465_ (.CLK(clk),
    .D(_045_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][0] ));
 sky130_fd_sc_hd__dfrtp_1 _466_ (.CLK(clk),
    .D(_046_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][1] ));
 sky130_fd_sc_hd__dfrtp_1 _467_ (.CLK(clk),
    .D(_047_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_reg[0][2] ));
 sky130_fd_sc_hd__dfrtp_1 _468_ (.CLK(clk),
    .D(_048_),
    .RESET_B(net17),
    .Q(net18));
 sky130_fd_sc_hd__dfrtp_1 _469_ (.CLK(clk),
    .D(_049_),
    .RESET_B(net17),
    .Q(net19));
 sky130_fd_sc_hd__dfrtp_1 _470_ (.CLK(clk),
    .D(_050_),
    .RESET_B(net17),
    .Q(net20));
 sky130_fd_sc_hd__dfrtp_1 _471_ (.CLK(clk),
    .D(_002_),
    .RESET_B(net17),
    .Q(net24));
 sky130_fd_sc_hd__dfrtp_1 _472_ (.CLK(clk),
    .D(_051_),
    .RESET_B(net17),
    .Q(net21));
 sky130_fd_sc_hd__dfrtp_1 _473_ (.CLK(clk),
    .D(_052_),
    .RESET_B(net17),
    .Q(net22));
 sky130_fd_sc_hd__dfrtp_1 _474_ (.CLK(clk),
    .D(_053_),
    .RESET_B(net17),
    .Q(net23));
 sky130_fd_sc_hd__dfrtp_4 _475_ (.CLK(clk),
    .D(_054_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[0] ));
 sky130_fd_sc_hd__dfrtp_1 _476_ (.CLK(clk),
    .D(_055_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[1] ));
 sky130_fd_sc_hd__dfrtp_1 _477_ (.CLK(clk),
    .D(_056_),
    .RESET_B(net17),
    .Q(\u_sort.bucket_counter[2] ));
 sky130_fd_sc_hd__dfrtp_1 _478_ (.CLK(clk),
    .D(_001_),
    .RESET_B(net17),
    .Q(\u_sort.busy ));
 sky130_fd_sc_hd__dfrtp_1 _479_ (.CLK(clk),
    .D(_057_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[0] ));
 sky130_fd_sc_hd__dfrtp_1 _480_ (.CLK(clk),
    .D(_058_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[1] ));
 sky130_fd_sc_hd__dfrtp_1 _481_ (.CLK(clk),
    .D(_059_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[2] ));
 sky130_fd_sc_hd__dfrtp_1 _482_ (.CLK(clk),
    .D(_060_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[3] ));
 sky130_fd_sc_hd__dfrtp_1 _483_ (.CLK(clk),
    .D(_061_),
    .RESET_B(net17),
    .Q(\u_sort.valid_reg[4] ));
 sky130_fd_sc_hd__dfrtp_4 _484_ (.CLK(clk),
    .D(_062_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[0] ));
 sky130_fd_sc_hd__dfrtp_4 _485_ (.CLK(clk),
    .D(_063_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[1] ));
 sky130_fd_sc_hd__dfrtp_4 _486_ (.CLK(clk),
    .D(_064_),
    .RESET_B(net17),
    .Q(\u_am0.i_index[2] ));
 sky130_fd_sc_hd__decap_3 PHY_0 ();
 sky130_fd_sc_hd__decap_3 PHY_1 ();
 sky130_fd_sc_hd__decap_3 PHY_2 ();
 sky130_fd_sc_hd__decap_3 PHY_3 ();
 sky130_fd_sc_hd__decap_3 PHY_4 ();
 sky130_fd_sc_hd__decap_3 PHY_5 ();
 sky130_fd_sc_hd__decap_3 PHY_6 ();
 sky130_fd_sc_hd__decap_3 PHY_7 ();
 sky130_fd_sc_hd__decap_3 PHY_8 ();
 sky130_fd_sc_hd__decap_3 PHY_9 ();
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
 sky130_fd_sc_hd__clkbuf_1 input1 (.A(i_valid),
    .X(net1));
 sky130_fd_sc_hd__clkbuf_1 input2 (.A(i_value0[0]),
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
 sky130_fd_sc_hd__buf_2 output18 (.A(net18),
    .X(o_bucket[0]));
 sky130_fd_sc_hd__buf_2 output19 (.A(net19),
    .X(o_bucket[1]));
 sky130_fd_sc_hd__buf_2 output20 (.A(net20),
    .X(o_bucket[2]));
 sky130_fd_sc_hd__clkbuf_4 output21 (.A(net21),
    .X(o_index[0]));
 sky130_fd_sc_hd__buf_2 output22 (.A(net22),
    .X(o_index[1]));
 sky130_fd_sc_hd__buf_2 output23 (.A(net23),
    .X(o_index[2]));
 sky130_fd_sc_hd__buf_2 output24 (.A(net24),
    .X(o_valid));
 sky130_fd_sc_hd__clkbuf_1 max_cap25 (.A(net26),
    .X(net25));
 sky130_fd_sc_hd__clkbuf_1 wire26 (.A(_103_),
    .X(net26));
endmodule
