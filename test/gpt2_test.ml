open Ctypes
open Gpt_2.C
open Model_explorer

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let attr key value = KeyValue.create ~key ~value
let pp_int64 fmt t = Format.fprintf fmt "%Ld" t
let pp_list p fmt t = Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@ ") p) t)
let pp_pair p1 p2 fmt (t1,t2) = Format.fprintf fmt "@[%a,@ %a@]" p1 t1 p2 t2

let pp_shape fmt t =
  let open Ggml.C in
  let rec cut_aux l' l =
    match l with [] -> l' | hd :: _ when hd = 1L || hd = 0L -> l' | hd :: tl -> cut_aux (hd :: l') tl
  in
  let ne = List.rev @@ cut_aux [] @@ CArray.to_list @@ getfp t Types.Tensor.ne in
  pp_list pp_int64 fmt ne

let pp_flags fmt t =
  let open Ggml.C in
  let flags = getfp t Types.Tensor.flags in
  let add name c l = if Int32.logand flags c = Int32.zero then l else name :: l in
  let flags =
    []
    |> add "Input" Ggml_const.C.Types.tensor_flag_input
    |> add "Output" Ggml_const.C.Types.tensor_flag_output
    |> add "Param" Ggml_const.C.Types.tensor_flag_param
    |> add "Loss" Ggml_const.C.Types.tensor_flag_loss
  in
  pp_list Format.pp_print_string fmt flags

module TensorId = struct
  module PtrMap = Map.Make (Nativeint)
  type kind =
  | Input
  | Output
  | Constant
  | Intermediate
  let kind_to_string kind = match  kind with
  | Input -> "Input"
  | Output -> "Output"
  | Constant -> "Constant"
  | Intermediate -> "Intermediate"

  type t = {
    id: int;
    kind: kind;
  }
  type nodes = {
    map: t PtrMap.t;
    node_count : int;
  }
  let empty node_count = {map=PtrMap.empty;node_count}

  let pp_addr fmt t = Format.fprintf fmt "%#LX" @@ Int64.of_nativeint t

  let pp fmt t =
    Format.fprintf fmt "@[{id:%d;@ kind:%s}"  t.id @@ kind_to_string t.kind

  let add_node id tensor nodes =
    assert (id < nodes.node_count);
    let open Ggml.C in
    let flags = getfp tensor Types.Tensor.flags in
    let kind = if Int32.logand flags Ggml_const.C.Types.tensor_flag_output = Int32.zero then Intermediate else Output in
    let t = {id;kind} in
    let ptr = Ctypes.raw_address_of_ptr @@ to_voidp tensor in
    {nodes with map=PtrMap.add ptr t nodes.map}

  let pp_nodes fmt t =
    let nodes = PtrMap.bindings t.map in
    Format.(pp_print_list ~pp_sep:pp_print_newline (pp_pair pp_addr pp)) fmt nodes
end

let tensor n t =
  let name = getfp t Ggml.C.Types.Tensor.name in
  let name = to_string @@ CArray.start name in
  let name =
    let printed_name =
      if String.length name == 0 || String.starts_with ~prefix:"leaf_" name || String.starts_with ~prefix:"node_" name
      then ""
      else name ^ " "
    in
    Printf.sprintf "%s(%s)" printed_name @@ Ggml.C.Functions.type_name @@ getfp t Ggml.C.Types.Tensor.typ_
  in
  let tensor_name = attr "tensor_name" name in
  let tensor_index = attr "tensor_index" @@ string_of_int n in
  let tensor_shape =
    let ne = getfp t Ggml.C.Types.Tensor.ne in
    let dims =
      if Ggml.C.Functions.is_matrix t then Printf.sprintf "[%Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1)
      else Printf.sprintf "[%Ld, %Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1) (CArray.get ne 2)
    in
    Printf.sprintf "%d %s %s" n dims (Ggml.C.Functions.op_symbol @@ getfp t Ggml.C.Types.Tensor.op)
  in
  ignore (tensor_name, tensor_index);
  ignore tensor_shape;
  ignore name;
  tensor_shape

let print pp t = Format.printf "@[%a@]@.%!" pp t

let%expect_test "gpt2" =
  let model = make Types.model_t in
  let n_ctx = 1024 in
  let n_gpu_layers = 0 in
  let fname = "models/gpt-2-117M/ggml-model.bin" in
  let rc = Functions.model_init (addr model) fname n_ctx n_gpu_layers in
  assert (rc = 0);
  [%expect {| gpt2_model_load: using CPU backend |}];

  let n_past = 0 in
  let n_tokens = 768 in

  let gpt2 = Functions.model_graph (addr model) n_past n_tokens in
  assert (not @@ Ctypes.is_null gpt2);
  let nodes = Ggml.C.Functions.graph_n_nodes gpt2 in
  Format.printf "nodes:%u" nodes;
  [%expect "nodes:487"];

  print pp_shape @@ Ggml.C.Functions.graph_node gpt2 0;
  [%expect "[768, 768]"];
  print pp_shape @@ Ggml.C.Functions.graph_node gpt2 6;
  [%expect "[2304, 768]"];

  print pp_flags @@ Ggml.C.Functions.graph_node gpt2 0;
  [%expect "[]"];
  print pp_flags @@ Ggml.C.Functions.graph_node gpt2 1;
  [%expect "[]"];
  print pp_flags @@ Ggml.C.Functions.graph_node gpt2 6;
  [%expect "[]"];
  print pp_flags @@ Ggml.C.Functions.graph_node gpt2 486;
  [%expect "[Output]"];

  let nodes = Array.init nodes (fun n -> Ggml.C.Functions.graph_node gpt2 n) in
  let tensors = ref @@ TensorId.empty @@ Array.length nodes in
  Array.iteri (fun id t -> tensors := TensorId.add_node id t !tensors) nodes;
  print TensorId.pp_nodes !tensors;
  [%expect
    {|
    0X7EFD933D2810, {id:1; kind:Intermediate}
    0X7EFD933D2980, {id:0; kind:Intermediate}
    0X7EFD933D2AF0, {id:2; kind:Intermediate}
    0X7EFD933D2C60, {id:3; kind:Intermediate}
    0X7EFD933D2DD0, {id:4; kind:Intermediate}
    0X7EFD933D2F40, {id:5; kind:Intermediate}
    0X7EFD933D30B0, {id:6; kind:Intermediate}
    0X7EFD933D3220, {id:7; kind:Intermediate}
    0X7EFD933D3390, {id:21; kind:Intermediate}
    0X7EFD933D3500, {id:8; kind:Intermediate}
    0X7EFD933D3670, {id:11; kind:Intermediate}
    0X7EFD933D37E0, {id:9; kind:Intermediate}
    0X7EFD933D3950, {id:12; kind:Intermediate}
    0X7EFD933D3AC0, {id:10; kind:Intermediate}
    0X7EFD933D3C30, {id:13; kind:Intermediate}
    0X7EFD933D3DA0, {id:22; kind:Intermediate}
    0X7EFD933D3F10, {id:23; kind:Intermediate}
    0X7EFD933D4080, {id:18; kind:Intermediate}
    0X7EFD933D41F0, {id:19; kind:Intermediate}
    0X7EFD933D4360, {id:20; kind:Intermediate}
    0X7EFD933D44D0, {id:24; kind:Intermediate}
    0X7EFD933D4640, {id:25; kind:Intermediate}
    0X7EFD933D47B0, {id:26; kind:Intermediate}
    0X7EFD933D4920, {id:27; kind:Intermediate}
    0X7EFD933D4A90, {id:14; kind:Intermediate}
    0X7EFD933D4C00, {id:15; kind:Intermediate}
    0X7EFD933D4D70, {id:16; kind:Intermediate}
    0X7EFD933D4EE0, {id:17; kind:Intermediate}
    0X7EFD933D5050, {id:28; kind:Intermediate}
    0X7EFD933D51C0, {id:29; kind:Intermediate}
    0X7EFD933D5330, {id:30; kind:Intermediate}
    0X7EFD933D54A0, {id:31; kind:Intermediate}
    0X7EFD933D5610, {id:32; kind:Intermediate}
    0X7EFD933D5780, {id:33; kind:Intermediate}
    0X7EFD933D58F0, {id:34; kind:Intermediate}
    0X7EFD933D5A60, {id:35; kind:Intermediate}
    0X7EFD933D5BD0, {id:36; kind:Intermediate}
    0X7EFD933D5D40, {id:37; kind:Intermediate}
    0X7EFD933D5EB0, {id:38; kind:Intermediate}
    0X7EFD933D6020, {id:39; kind:Intermediate}
    0X7EFD933D6190, {id:40; kind:Intermediate}
    0X7EFD933D6300, {id:41; kind:Intermediate}
    0X7EFD933D6470, {id:42; kind:Intermediate}
    0X7EFD933D65E0, {id:43; kind:Intermediate}
    0X7EFD933D6750, {id:44; kind:Intermediate}
    0X7EFD933D68C0, {id:45; kind:Intermediate}
    0X7EFD933D6A30, {id:46; kind:Intermediate}
    0X7EFD933D6BA0, {id:47; kind:Intermediate}
    0X7EFD933D6D10, {id:61; kind:Intermediate}
    0X7EFD933D6E80, {id:48; kind:Intermediate}
    0X7EFD933D6FF0, {id:51; kind:Intermediate}
    0X7EFD933D7160, {id:49; kind:Intermediate}
    0X7EFD933D72D0, {id:52; kind:Intermediate}
    0X7EFD933D7440, {id:50; kind:Intermediate}
    0X7EFD933D75B0, {id:53; kind:Intermediate}
    0X7EFD933D7720, {id:62; kind:Intermediate}
    0X7EFD933D7890, {id:63; kind:Intermediate}
    0X7EFD933D7A00, {id:58; kind:Intermediate}
    0X7EFD933D7B70, {id:59; kind:Intermediate}
    0X7EFD933D7CE0, {id:60; kind:Intermediate}
    0X7EFD933D7E50, {id:64; kind:Intermediate}
    0X7EFD933D7FC0, {id:65; kind:Intermediate}
    0X7EFD933D8130, {id:66; kind:Intermediate}
    0X7EFD933D82A0, {id:67; kind:Intermediate}
    0X7EFD933D8410, {id:54; kind:Intermediate}
    0X7EFD933D8580, {id:55; kind:Intermediate}
    0X7EFD933D86F0, {id:56; kind:Intermediate}
    0X7EFD933D8860, {id:57; kind:Intermediate}
    0X7EFD933D89D0, {id:68; kind:Intermediate}
    0X7EFD933D8B40, {id:69; kind:Intermediate}
    0X7EFD933D8CB0, {id:70; kind:Intermediate}
    0X7EFD933D8E20, {id:71; kind:Intermediate}
    0X7EFD933D8F90, {id:72; kind:Intermediate}
    0X7EFD933D9100, {id:73; kind:Intermediate}
    0X7EFD933D9270, {id:74; kind:Intermediate}
    0X7EFD933D93E0, {id:75; kind:Intermediate}
    0X7EFD933D9550, {id:76; kind:Intermediate}
    0X7EFD933D96C0, {id:77; kind:Intermediate}
    0X7EFD933D9830, {id:78; kind:Intermediate}
    0X7EFD933D99A0, {id:79; kind:Intermediate}
    0X7EFD933D9B10, {id:80; kind:Intermediate}
    0X7EFD933D9C80, {id:81; kind:Intermediate}
    0X7EFD933D9DF0, {id:82; kind:Intermediate}
    0X7EFD933D9F60, {id:83; kind:Intermediate}
    0X7EFD933DA0D0, {id:84; kind:Intermediate}
    0X7EFD933DA240, {id:85; kind:Intermediate}
    0X7EFD933DA3B0, {id:86; kind:Intermediate}
    0X7EFD933DA520, {id:87; kind:Intermediate}
    0X7EFD933DA690, {id:101; kind:Intermediate}
    0X7EFD933DA800, {id:88; kind:Intermediate}
    0X7EFD933DA970, {id:91; kind:Intermediate}
    0X7EFD933DAAE0, {id:89; kind:Intermediate}
    0X7EFD933DAC50, {id:92; kind:Intermediate}
    0X7EFD933DADC0, {id:90; kind:Intermediate}
    0X7EFD933DAF30, {id:93; kind:Intermediate}
    0X7EFD933DB0A0, {id:102; kind:Intermediate}
    0X7EFD933DB210, {id:103; kind:Intermediate}
    0X7EFD933DB380, {id:98; kind:Intermediate}
    0X7EFD933DB4F0, {id:99; kind:Intermediate}
    0X7EFD933DB660, {id:100; kind:Intermediate}
    0X7EFD933DB7D0, {id:104; kind:Intermediate}
    0X7EFD933DB940, {id:105; kind:Intermediate}
    0X7EFD933DBAB0, {id:106; kind:Intermediate}
    0X7EFD933DBC20, {id:107; kind:Intermediate}
    0X7EFD933DBD90, {id:94; kind:Intermediate}
    0X7EFD933DBF00, {id:95; kind:Intermediate}
    0X7EFD933DC070, {id:96; kind:Intermediate}
    0X7EFD933DC1E0, {id:97; kind:Intermediate}
    0X7EFD933DC350, {id:108; kind:Intermediate}
    0X7EFD933DC4C0, {id:109; kind:Intermediate}
    0X7EFD933DC630, {id:110; kind:Intermediate}
    0X7EFD933DC7A0, {id:111; kind:Intermediate}
    0X7EFD933DC910, {id:112; kind:Intermediate}
    0X7EFD933DCA80, {id:113; kind:Intermediate}
    0X7EFD933DCBF0, {id:114; kind:Intermediate}
    0X7EFD933DCD60, {id:115; kind:Intermediate}
    0X7EFD933DCED0, {id:116; kind:Intermediate}
    0X7EFD933DD040, {id:117; kind:Intermediate}
    0X7EFD933DD1B0, {id:118; kind:Intermediate}
    0X7EFD933DD320, {id:119; kind:Intermediate}
    0X7EFD933DD490, {id:120; kind:Intermediate}
    0X7EFD933DD600, {id:121; kind:Intermediate}
    0X7EFD933DD770, {id:122; kind:Intermediate}
    0X7EFD933DD8E0, {id:123; kind:Intermediate}
    0X7EFD933DDA50, {id:124; kind:Intermediate}
    0X7EFD933DDBC0, {id:125; kind:Intermediate}
    0X7EFD933DDD30, {id:126; kind:Intermediate}
    0X7EFD933DDEA0, {id:127; kind:Intermediate}
    0X7EFD933DE010, {id:141; kind:Intermediate}
    0X7EFD933DE180, {id:128; kind:Intermediate}
    0X7EFD933DE2F0, {id:131; kind:Intermediate}
    0X7EFD933DE460, {id:129; kind:Intermediate}
    0X7EFD933DE5D0, {id:132; kind:Intermediate}
    0X7EFD933DE740, {id:130; kind:Intermediate}
    0X7EFD933DE8B0, {id:133; kind:Intermediate}
    0X7EFD933DEA20, {id:142; kind:Intermediate}
    0X7EFD933DEB90, {id:143; kind:Intermediate}
    0X7EFD933DED00, {id:138; kind:Intermediate}
    0X7EFD933DEE70, {id:139; kind:Intermediate}
    0X7EFD933DEFE0, {id:140; kind:Intermediate}
    0X7EFD933DF150, {id:144; kind:Intermediate}
    0X7EFD933DF2C0, {id:145; kind:Intermediate}
    0X7EFD933DF430, {id:146; kind:Intermediate}
    0X7EFD933DF5A0, {id:147; kind:Intermediate}
    0X7EFD933DF710, {id:134; kind:Intermediate}
    0X7EFD933DF880, {id:135; kind:Intermediate}
    0X7EFD933DF9F0, {id:136; kind:Intermediate}
    0X7EFD933DFB60, {id:137; kind:Intermediate}
    0X7EFD933DFCD0, {id:148; kind:Intermediate}
    0X7EFD933DFE40, {id:149; kind:Intermediate}
    0X7EFD933DFFB0, {id:150; kind:Intermediate}
    0X7EFD933E0120, {id:151; kind:Intermediate}
    0X7EFD933E0290, {id:152; kind:Intermediate}
    0X7EFD933E0400, {id:153; kind:Intermediate}
    0X7EFD933E0570, {id:154; kind:Intermediate}
    0X7EFD933E06E0, {id:155; kind:Intermediate}
    0X7EFD933E0850, {id:156; kind:Intermediate}
    0X7EFD933E09C0, {id:157; kind:Intermediate}
    0X7EFD933E0B30, {id:158; kind:Intermediate}
    0X7EFD933E0CA0, {id:159; kind:Intermediate}
    0X7EFD933E0E10, {id:160; kind:Intermediate}
    0X7EFD933E0F80, {id:161; kind:Intermediate}
    0X7EFD933E10F0, {id:162; kind:Intermediate}
    0X7EFD933E1260, {id:163; kind:Intermediate}
    0X7EFD933E13D0, {id:164; kind:Intermediate}
    0X7EFD933E1540, {id:165; kind:Intermediate}
    0X7EFD933E16B0, {id:166; kind:Intermediate}
    0X7EFD933E1820, {id:167; kind:Intermediate}
    0X7EFD933E1990, {id:181; kind:Intermediate}
    0X7EFD933E1B00, {id:168; kind:Intermediate}
    0X7EFD933E1C70, {id:171; kind:Intermediate}
    0X7EFD933E1DE0, {id:169; kind:Intermediate}
    0X7EFD933E1F50, {id:172; kind:Intermediate}
    0X7EFD933E20C0, {id:170; kind:Intermediate}
    0X7EFD933E2230, {id:173; kind:Intermediate}
    0X7EFD933E23A0, {id:182; kind:Intermediate}
    0X7EFD933E2510, {id:183; kind:Intermediate}
    0X7EFD933E2680, {id:178; kind:Intermediate}
    0X7EFD933E27F0, {id:179; kind:Intermediate}
    0X7EFD933E2960, {id:180; kind:Intermediate}
    0X7EFD933E2AD0, {id:184; kind:Intermediate}
    0X7EFD933E2C40, {id:185; kind:Intermediate}
    0X7EFD933E2DB0, {id:186; kind:Intermediate}
    0X7EFD933E2F20, {id:187; kind:Intermediate}
    0X7EFD933E3090, {id:174; kind:Intermediate}
    0X7EFD933E3200, {id:175; kind:Intermediate}
    0X7EFD933E3370, {id:176; kind:Intermediate}
    0X7EFD933E34E0, {id:177; kind:Intermediate}
    0X7EFD933E3650, {id:188; kind:Intermediate}
    0X7EFD933E37C0, {id:189; kind:Intermediate}
    0X7EFD933E3930, {id:190; kind:Intermediate}
    0X7EFD933E3AA0, {id:191; kind:Intermediate}
    0X7EFD933E3C10, {id:192; kind:Intermediate}
    0X7EFD933E3D80, {id:193; kind:Intermediate}
    0X7EFD933E3EF0, {id:194; kind:Intermediate}
    0X7EFD933E4060, {id:195; kind:Intermediate}
    0X7EFD933E41D0, {id:196; kind:Intermediate}
    0X7EFD933E4340, {id:197; kind:Intermediate}
    0X7EFD933E44B0, {id:198; kind:Intermediate}
    0X7EFD933E4620, {id:199; kind:Intermediate}
    0X7EFD933E4790, {id:200; kind:Intermediate}
    0X7EFD933E4900, {id:201; kind:Intermediate}
    0X7EFD933E4A70, {id:202; kind:Intermediate}
    0X7EFD933E4BE0, {id:203; kind:Intermediate}
    0X7EFD933E4D50, {id:204; kind:Intermediate}
    0X7EFD933E4EC0, {id:205; kind:Intermediate}
    0X7EFD933E5030, {id:206; kind:Intermediate}
    0X7EFD933E51A0, {id:207; kind:Intermediate}
    0X7EFD933E5310, {id:221; kind:Intermediate}
    0X7EFD933E5480, {id:208; kind:Intermediate}
    0X7EFD933E55F0, {id:211; kind:Intermediate}
    0X7EFD933E5760, {id:209; kind:Intermediate}
    0X7EFD933E58D0, {id:212; kind:Intermediate}
    0X7EFD933E5A40, {id:210; kind:Intermediate}
    0X7EFD933E5BB0, {id:213; kind:Intermediate}
    0X7EFD933E5D20, {id:222; kind:Intermediate}
    0X7EFD933E5E90, {id:223; kind:Intermediate}
    0X7EFD933E6000, {id:218; kind:Intermediate}
    0X7EFD933E6170, {id:219; kind:Intermediate}
    0X7EFD933E62E0, {id:220; kind:Intermediate}
    0X7EFD933E6450, {id:224; kind:Intermediate}
    0X7EFD933E65C0, {id:225; kind:Intermediate}
    0X7EFD933E6730, {id:226; kind:Intermediate}
    0X7EFD933E68A0, {id:227; kind:Intermediate}
    0X7EFD933E6A10, {id:214; kind:Intermediate}
    0X7EFD933E6B80, {id:215; kind:Intermediate}
    0X7EFD933E6CF0, {id:216; kind:Intermediate}
    0X7EFD933E6E60, {id:217; kind:Intermediate}
    0X7EFD933E6FD0, {id:228; kind:Intermediate}
    0X7EFD933E7140, {id:229; kind:Intermediate}
    0X7EFD933E72B0, {id:230; kind:Intermediate}
    0X7EFD933E7420, {id:231; kind:Intermediate}
    0X7EFD933E7590, {id:232; kind:Intermediate}
    0X7EFD933E7700, {id:233; kind:Intermediate}
    0X7EFD933E7870, {id:234; kind:Intermediate}
    0X7EFD933E79E0, {id:235; kind:Intermediate}
    0X7EFD933E7B50, {id:236; kind:Intermediate}
    0X7EFD933E7CC0, {id:237; kind:Intermediate}
    0X7EFD933E7E30, {id:238; kind:Intermediate}
    0X7EFD933E7FA0, {id:239; kind:Intermediate}
    0X7EFD933E8110, {id:240; kind:Intermediate}
    0X7EFD933E8280, {id:241; kind:Intermediate}
    0X7EFD933E83F0, {id:242; kind:Intermediate}
    0X7EFD933E8560, {id:243; kind:Intermediate}
    0X7EFD933E86D0, {id:244; kind:Intermediate}
    0X7EFD933E8840, {id:245; kind:Intermediate}
    0X7EFD933E89B0, {id:246; kind:Intermediate}
    0X7EFD933E8B20, {id:247; kind:Intermediate}
    0X7EFD933E8C90, {id:261; kind:Intermediate}
    0X7EFD933E8E00, {id:248; kind:Intermediate}
    0X7EFD933E8F70, {id:251; kind:Intermediate}
    0X7EFD933E90E0, {id:249; kind:Intermediate}
    0X7EFD933E9250, {id:252; kind:Intermediate}
    0X7EFD933E93C0, {id:250; kind:Intermediate}
    0X7EFD933E9530, {id:253; kind:Intermediate}
    0X7EFD933E96A0, {id:262; kind:Intermediate}
    0X7EFD933E9810, {id:263; kind:Intermediate}
    0X7EFD933E9980, {id:258; kind:Intermediate}
    0X7EFD933E9AF0, {id:259; kind:Intermediate}
    0X7EFD933E9C60, {id:260; kind:Intermediate}
    0X7EFD933E9DD0, {id:264; kind:Intermediate}
    0X7EFD933E9F40, {id:265; kind:Intermediate}
    0X7EFD933EA0B0, {id:266; kind:Intermediate}
    0X7EFD933EA220, {id:267; kind:Intermediate}
    0X7EFD933EA390, {id:254; kind:Intermediate}
    0X7EFD933EA500, {id:255; kind:Intermediate}
    0X7EFD933EA670, {id:256; kind:Intermediate}
    0X7EFD933EA7E0, {id:257; kind:Intermediate}
    0X7EFD933EA950, {id:268; kind:Intermediate}
    0X7EFD933EAAC0, {id:269; kind:Intermediate}
    0X7EFD933EAC30, {id:270; kind:Intermediate}
    0X7EFD933EADA0, {id:271; kind:Intermediate}
    0X7EFD933EAF10, {id:272; kind:Intermediate}
    0X7EFD933EB080, {id:273; kind:Intermediate}
    0X7EFD933EB1F0, {id:274; kind:Intermediate}
    0X7EFD933EB360, {id:275; kind:Intermediate}
    0X7EFD933EB4D0, {id:276; kind:Intermediate}
    0X7EFD933EB640, {id:277; kind:Intermediate}
    0X7EFD933EB7B0, {id:278; kind:Intermediate}
    0X7EFD933EB920, {id:279; kind:Intermediate}
    0X7EFD933EBA90, {id:280; kind:Intermediate}
    0X7EFD933EBC00, {id:281; kind:Intermediate}
    0X7EFD933EBD70, {id:282; kind:Intermediate}
    0X7EFD933EBEE0, {id:283; kind:Intermediate}
    0X7EFD933EC050, {id:284; kind:Intermediate}
    0X7EFD933EC1C0, {id:285; kind:Intermediate}
    0X7EFD933EC330, {id:286; kind:Intermediate}
    0X7EFD933EC4A0, {id:287; kind:Intermediate}
    0X7EFD933EC610, {id:301; kind:Intermediate}
    0X7EFD933EC780, {id:288; kind:Intermediate}
    0X7EFD933EC8F0, {id:291; kind:Intermediate}
    0X7EFD933ECA60, {id:289; kind:Intermediate}
    0X7EFD933ECBD0, {id:292; kind:Intermediate}
    0X7EFD933ECD40, {id:290; kind:Intermediate}
    0X7EFD933ECEB0, {id:293; kind:Intermediate}
    0X7EFD933ED020, {id:302; kind:Intermediate}
    0X7EFD933ED190, {id:303; kind:Intermediate}
    0X7EFD933ED300, {id:298; kind:Intermediate}
    0X7EFD933ED470, {id:299; kind:Intermediate}
    0X7EFD933ED5E0, {id:300; kind:Intermediate}
    0X7EFD933ED750, {id:304; kind:Intermediate}
    0X7EFD933ED8C0, {id:305; kind:Intermediate}
    0X7EFD933EDA30, {id:306; kind:Intermediate}
    0X7EFD933EDBA0, {id:307; kind:Intermediate}
    0X7EFD933EDD10, {id:294; kind:Intermediate}
    0X7EFD933EDE80, {id:295; kind:Intermediate}
    0X7EFD933EDFF0, {id:296; kind:Intermediate}
    0X7EFD933EE160, {id:297; kind:Intermediate}
    0X7EFD933EE2D0, {id:308; kind:Intermediate}
    0X7EFD933EE440, {id:309; kind:Intermediate}
    0X7EFD933EE5B0, {id:310; kind:Intermediate}
    0X7EFD933EE720, {id:311; kind:Intermediate}
    0X7EFD933EE890, {id:312; kind:Intermediate}
    0X7EFD933EEA00, {id:313; kind:Intermediate}
    0X7EFD933EEB70, {id:314; kind:Intermediate}
    0X7EFD933EECE0, {id:315; kind:Intermediate}
    0X7EFD933EEE50, {id:316; kind:Intermediate}
    0X7EFD933EEFC0, {id:317; kind:Intermediate}
    0X7EFD933EF130, {id:318; kind:Intermediate}
    0X7EFD933EF2A0, {id:319; kind:Intermediate}
    0X7EFD933EF410, {id:320; kind:Intermediate}
    0X7EFD933EF580, {id:321; kind:Intermediate}
    0X7EFD933EF6F0, {id:322; kind:Intermediate}
    0X7EFD933EF860, {id:323; kind:Intermediate}
    0X7EFD933EF9D0, {id:324; kind:Intermediate}
    0X7EFD933EFB40, {id:325; kind:Intermediate}
    0X7EFD933EFCB0, {id:326; kind:Intermediate}
    0X7EFD933EFE20, {id:327; kind:Intermediate}
    0X7EFD933EFF90, {id:341; kind:Intermediate}
    0X7EFD933F0100, {id:328; kind:Intermediate}
    0X7EFD933F0270, {id:331; kind:Intermediate}
    0X7EFD933F03E0, {id:329; kind:Intermediate}
    0X7EFD933F0550, {id:332; kind:Intermediate}
    0X7EFD933F06C0, {id:330; kind:Intermediate}
    0X7EFD933F0830, {id:333; kind:Intermediate}
    0X7EFD933F09A0, {id:342; kind:Intermediate}
    0X7EFD933F0B10, {id:343; kind:Intermediate}
    0X7EFD933F0C80, {id:338; kind:Intermediate}
    0X7EFD933F0DF0, {id:339; kind:Intermediate}
    0X7EFD933F0F60, {id:340; kind:Intermediate}
    0X7EFD933F10D0, {id:344; kind:Intermediate}
    0X7EFD933F1240, {id:345; kind:Intermediate}
    0X7EFD933F13B0, {id:346; kind:Intermediate}
    0X7EFD933F1520, {id:347; kind:Intermediate}
    0X7EFD933F1690, {id:334; kind:Intermediate}
    0X7EFD933F1800, {id:335; kind:Intermediate}
    0X7EFD933F1970, {id:336; kind:Intermediate}
    0X7EFD933F1AE0, {id:337; kind:Intermediate}
    0X7EFD933F1C50, {id:348; kind:Intermediate}
    0X7EFD933F1DC0, {id:349; kind:Intermediate}
    0X7EFD933F1F30, {id:350; kind:Intermediate}
    0X7EFD933F20A0, {id:351; kind:Intermediate}
    0X7EFD933F2210, {id:352; kind:Intermediate}
    0X7EFD933F2380, {id:353; kind:Intermediate}
    0X7EFD933F24F0, {id:354; kind:Intermediate}
    0X7EFD933F2660, {id:355; kind:Intermediate}
    0X7EFD933F27D0, {id:356; kind:Intermediate}
    0X7EFD933F2940, {id:357; kind:Intermediate}
    0X7EFD933F2AB0, {id:358; kind:Intermediate}
    0X7EFD933F2C20, {id:359; kind:Intermediate}
    0X7EFD933F2D90, {id:360; kind:Intermediate}
    0X7EFD933F2F00, {id:361; kind:Intermediate}
    0X7EFD933F3070, {id:362; kind:Intermediate}
    0X7EFD933F31E0, {id:363; kind:Intermediate}
    0X7EFD933F3350, {id:364; kind:Intermediate}
    0X7EFD933F34C0, {id:365; kind:Intermediate}
    0X7EFD933F3630, {id:366; kind:Intermediate}
    0X7EFD933F37A0, {id:367; kind:Intermediate}
    0X7EFD933F3910, {id:381; kind:Intermediate}
    0X7EFD933F3A80, {id:368; kind:Intermediate}
    0X7EFD933F3BF0, {id:371; kind:Intermediate}
    0X7EFD933F3D60, {id:369; kind:Intermediate}
    0X7EFD933F3ED0, {id:372; kind:Intermediate}
    0X7EFD933F4040, {id:370; kind:Intermediate}
    0X7EFD933F41B0, {id:373; kind:Intermediate}
    0X7EFD933F4320, {id:382; kind:Intermediate}
    0X7EFD933F4490, {id:383; kind:Intermediate}
    0X7EFD933F4600, {id:378; kind:Intermediate}
    0X7EFD933F4770, {id:379; kind:Intermediate}
    0X7EFD933F48E0, {id:380; kind:Intermediate}
    0X7EFD933F4A50, {id:384; kind:Intermediate}
    0X7EFD933F4BC0, {id:385; kind:Intermediate}
    0X7EFD933F4D30, {id:386; kind:Intermediate}
    0X7EFD933F4EA0, {id:387; kind:Intermediate}
    0X7EFD933F5010, {id:374; kind:Intermediate}
    0X7EFD933F5180, {id:375; kind:Intermediate}
    0X7EFD933F52F0, {id:376; kind:Intermediate}
    0X7EFD933F5460, {id:377; kind:Intermediate}
    0X7EFD933F55D0, {id:388; kind:Intermediate}
    0X7EFD933F5740, {id:389; kind:Intermediate}
    0X7EFD933F58B0, {id:390; kind:Intermediate}
    0X7EFD933F5A20, {id:391; kind:Intermediate}
    0X7EFD933F5B90, {id:392; kind:Intermediate}
    0X7EFD933F5D00, {id:393; kind:Intermediate}
    0X7EFD933F5E70, {id:394; kind:Intermediate}
    0X7EFD933F5FE0, {id:395; kind:Intermediate}
    0X7EFD933F6150, {id:396; kind:Intermediate}
    0X7EFD933F62C0, {id:397; kind:Intermediate}
    0X7EFD933F6430, {id:398; kind:Intermediate}
    0X7EFD933F65A0, {id:399; kind:Intermediate}
    0X7EFD933F6710, {id:400; kind:Intermediate}
    0X7EFD933F6880, {id:401; kind:Intermediate}
    0X7EFD933F69F0, {id:402; kind:Intermediate}
    0X7EFD933F6B60, {id:403; kind:Intermediate}
    0X7EFD933F6CD0, {id:404; kind:Intermediate}
    0X7EFD933F6E40, {id:405; kind:Intermediate}
    0X7EFD933F6FB0, {id:406; kind:Intermediate}
    0X7EFD933F7120, {id:407; kind:Intermediate}
    0X7EFD933F7290, {id:421; kind:Intermediate}
    0X7EFD933F7400, {id:408; kind:Intermediate}
    0X7EFD933F7570, {id:411; kind:Intermediate}
    0X7EFD933F76E0, {id:409; kind:Intermediate}
    0X7EFD933F7850, {id:412; kind:Intermediate}
    0X7EFD933F79C0, {id:410; kind:Intermediate}
    0X7EFD933F7B30, {id:413; kind:Intermediate}
    0X7EFD933F7CA0, {id:422; kind:Intermediate}
    0X7EFD933F7E10, {id:423; kind:Intermediate}
    0X7EFD933F7F80, {id:418; kind:Intermediate}
    0X7EFD933F80F0, {id:419; kind:Intermediate}
    0X7EFD933F8260, {id:420; kind:Intermediate}
    0X7EFD933F83D0, {id:424; kind:Intermediate}
    0X7EFD933F8540, {id:425; kind:Intermediate}
    0X7EFD933F86B0, {id:426; kind:Intermediate}
    0X7EFD933F8820, {id:427; kind:Intermediate}
    0X7EFD933F8990, {id:414; kind:Intermediate}
    0X7EFD933F8B00, {id:415; kind:Intermediate}
    0X7EFD933F8C70, {id:416; kind:Intermediate}
    0X7EFD933F8DE0, {id:417; kind:Intermediate}
    0X7EFD933F8F50, {id:428; kind:Intermediate}
    0X7EFD933F90C0, {id:429; kind:Intermediate}
    0X7EFD933F9230, {id:430; kind:Intermediate}
    0X7EFD933F93A0, {id:431; kind:Intermediate}
    0X7EFD933F9510, {id:432; kind:Intermediate}
    0X7EFD933F9680, {id:433; kind:Intermediate}
    0X7EFD933F97F0, {id:434; kind:Intermediate}
    0X7EFD933F9960, {id:435; kind:Intermediate}
    0X7EFD933F9AD0, {id:436; kind:Intermediate}
    0X7EFD933F9C40, {id:437; kind:Intermediate}
    0X7EFD933F9DB0, {id:438; kind:Intermediate}
    0X7EFD933F9F20, {id:439; kind:Intermediate}
    0X7EFD933FA090, {id:440; kind:Intermediate}
    0X7EFD933FA200, {id:441; kind:Intermediate}
    0X7EFD933FA370, {id:442; kind:Intermediate}
    0X7EFD933FA4E0, {id:443; kind:Intermediate}
    0X7EFD933FA650, {id:444; kind:Intermediate}
    0X7EFD933FA7C0, {id:445; kind:Intermediate}
    0X7EFD933FA930, {id:446; kind:Intermediate}
    0X7EFD933FAAA0, {id:447; kind:Intermediate}
    0X7EFD933FAC10, {id:461; kind:Intermediate}
    0X7EFD933FAD80, {id:448; kind:Intermediate}
    0X7EFD933FAEF0, {id:451; kind:Intermediate}
    0X7EFD933FB060, {id:449; kind:Intermediate}
    0X7EFD933FB1D0, {id:452; kind:Intermediate}
    0X7EFD933FB340, {id:450; kind:Intermediate}
    0X7EFD933FB4B0, {id:453; kind:Intermediate}
    0X7EFD933FB620, {id:462; kind:Intermediate}
    0X7EFD933FB790, {id:463; kind:Intermediate}
    0X7EFD933FB900, {id:458; kind:Intermediate}
    0X7EFD933FBA70, {id:459; kind:Intermediate}
    0X7EFD933FBBE0, {id:460; kind:Intermediate}
    0X7EFD933FBD50, {id:464; kind:Intermediate}
    0X7EFD933FBEC0, {id:465; kind:Intermediate}
    0X7EFD933FC030, {id:466; kind:Intermediate}
    0X7EFD933FC1A0, {id:467; kind:Intermediate}
    0X7EFD933FC310, {id:454; kind:Intermediate}
    0X7EFD933FC480, {id:455; kind:Intermediate}
    0X7EFD933FC5F0, {id:456; kind:Intermediate}
    0X7EFD933FC760, {id:457; kind:Intermediate}
    0X7EFD933FC8D0, {id:468; kind:Intermediate}
    0X7EFD933FCA40, {id:469; kind:Intermediate}
    0X7EFD933FCBB0, {id:470; kind:Intermediate}
    0X7EFD933FCD20, {id:471; kind:Intermediate}
    0X7EFD933FCE90, {id:472; kind:Intermediate}
    0X7EFD933FD000, {id:473; kind:Intermediate}
    0X7EFD933FD170, {id:474; kind:Intermediate}
    0X7EFD933FD2E0, {id:475; kind:Intermediate}
    0X7EFD933FD450, {id:476; kind:Intermediate}
    0X7EFD933FD5C0, {id:477; kind:Intermediate}
    0X7EFD933FD730, {id:478; kind:Intermediate}
    0X7EFD933FD8A0, {id:479; kind:Intermediate}
    0X7EFD933FDA10, {id:480; kind:Intermediate}
    0X7EFD933FDB80, {id:481; kind:Intermediate}
    0X7EFD933FDCF0, {id:482; kind:Intermediate}
    0X7EFD933FDE60, {id:483; kind:Intermediate}
    0X7EFD933FDFD0, {id:484; kind:Intermediate}
    0X7EFD933FE140, {id:485; kind:Intermediate}
    0X7EFD933FE2B0, {id:486; kind:Output} |}];

  let names = Array.map (fun t -> Ggml.C.Functions.op_name @@ getfp t Ggml.C.Types.Tensor.op) nodes in
  ignore names;
  let names = Array.mapi tensor nodes in

  Format.printf "@[%a@]" (Format.pp_print_list ~pp_sep:Format.pp_print_newline Format.pp_print_string)
  @@ Array.to_list names;
  [%expect
    {|
    0 [768, 768] get_rows(x)
    1 [768, 768] get_rows(x)
    2 [768, 768] x+y
    3 [768, 768] norm(x)
    4 [768, 768] x*y
    5 [768, 768] x+y
    6 [2304, 768] X*Y
    7 [2304, 768] x+y
    8 [768, 768] view(x)
    9 [589824, 1] view(x)
    10 [589824, 1] x-\>y
    11 [768, 768] view(x)
    12 [589824, 1] view(x)
    13 [589824, 1] x-\>y
    14 [589824, 1] view(x)
    15 [64, 12, 768] reshape(x)
    16 [768, 64, 12] permute(x)
    17 [768, 64, 12] cont(x)
    18 [589824, 1] view(x)
    19 [64, 12, 768] reshape(x)
    20 [64, 768, 12] permute(x)
    21 [768, 768] view(x)
    22 [64, 12, 768] cont(x)
    23 [64, 768, 12] permute(x)
    24 [768, 768, 12] X*Y
    25 [768, 768, 12] x*v
    26 [768, 768, 12] diag_mask_inf(x)
    27 [768, 768, 12] soft_max(x)
    28 [64, 768, 12] X*Y
    29 [64, 12, 768] permute(x)
    30 [768, 768] cont(x)
    31 [768, 768] X*Y
    32 [768, 768] x+y
    33 [768, 768] x+y
    34 [768, 768] norm(x)
    35 [768, 768] x*y
    36 [768, 768] x+y
    37 [3072, 768] X*Y
    38 [3072, 768] x+y
    39 [3072, 768] unary(x)
    40 [768, 768] X*Y
    41 [768, 768] x+y
    42 [768, 768] x+y
    43 [768, 768] norm(x)
    44 [768, 768] x*y
    45 [768, 768] x+y
    46 [2304, 768] X*Y
    47 [2304, 768] x+y
    48 [768, 768] view(x)
    49 [589824, 1] view(x)
    50 [589824, 1] x-\>y
    51 [768, 768] view(x)
    52 [589824, 1] view(x)
    53 [589824, 1] x-\>y
    54 [589824, 1] view(x)
    55 [64, 12, 768] reshape(x)
    56 [768, 64, 12] permute(x)
    57 [768, 64, 12] cont(x)
    58 [589824, 1] view(x)
    59 [64, 12, 768] reshape(x)
    60 [64, 768, 12] permute(x)
    61 [768, 768] view(x)
    62 [64, 12, 768] cont(x)
    63 [64, 768, 12] permute(x)
    64 [768, 768, 12] X*Y
    65 [768, 768, 12] x*v
    66 [768, 768, 12] diag_mask_inf(x)
    67 [768, 768, 12] soft_max(x)
    68 [64, 768, 12] X*Y
    69 [64, 12, 768] permute(x)
    70 [768, 768] cont(x)
    71 [768, 768] X*Y
    72 [768, 768] x+y
    73 [768, 768] x+y
    74 [768, 768] norm(x)
    75 [768, 768] x*y
    76 [768, 768] x+y
    77 [3072, 768] X*Y
    78 [3072, 768] x+y
    79 [3072, 768] unary(x)
    80 [768, 768] X*Y
    81 [768, 768] x+y
    82 [768, 768] x+y
    83 [768, 768] norm(x)
    84 [768, 768] x*y
    85 [768, 768] x+y
    86 [2304, 768] X*Y
    87 [2304, 768] x+y
    88 [768, 768] view(x)
    89 [589824, 1] view(x)
    90 [589824, 1] x-\>y
    91 [768, 768] view(x)
    92 [589824, 1] view(x)
    93 [589824, 1] x-\>y
    94 [589824, 1] view(x)
    95 [64, 12, 768] reshape(x)
    96 [768, 64, 12] permute(x)
    97 [768, 64, 12] cont(x)
    98 [589824, 1] view(x)
    99 [64, 12, 768] reshape(x)
    100 [64, 768, 12] permute(x)
    101 [768, 768] view(x)
    102 [64, 12, 768] cont(x)
    103 [64, 768, 12] permute(x)
    104 [768, 768, 12] X*Y
    105 [768, 768, 12] x*v
    106 [768, 768, 12] diag_mask_inf(x)
    107 [768, 768, 12] soft_max(x)
    108 [64, 768, 12] X*Y
    109 [64, 12, 768] permute(x)
    110 [768, 768] cont(x)
    111 [768, 768] X*Y
    112 [768, 768] x+y
    113 [768, 768] x+y
    114 [768, 768] norm(x)
    115 [768, 768] x*y
    116 [768, 768] x+y
    117 [3072, 768] X*Y
    118 [3072, 768] x+y
    119 [3072, 768] unary(x)
    120 [768, 768] X*Y
    121 [768, 768] x+y
    122 [768, 768] x+y
    123 [768, 768] norm(x)
    124 [768, 768] x*y
    125 [768, 768] x+y
    126 [2304, 768] X*Y
    127 [2304, 768] x+y
    128 [768, 768] view(x)
    129 [589824, 1] view(x)
    130 [589824, 1] x-\>y
    131 [768, 768] view(x)
    132 [589824, 1] view(x)
    133 [589824, 1] x-\>y
    134 [589824, 1] view(x)
    135 [64, 12, 768] reshape(x)
    136 [768, 64, 12] permute(x)
    137 [768, 64, 12] cont(x)
    138 [589824, 1] view(x)
    139 [64, 12, 768] reshape(x)
    140 [64, 768, 12] permute(x)
    141 [768, 768] view(x)
    142 [64, 12, 768] cont(x)
    143 [64, 768, 12] permute(x)
    144 [768, 768, 12] X*Y
    145 [768, 768, 12] x*v
    146 [768, 768, 12] diag_mask_inf(x)
    147 [768, 768, 12] soft_max(x)
    148 [64, 768, 12] X*Y
    149 [64, 12, 768] permute(x)
    150 [768, 768] cont(x)
    151 [768, 768] X*Y
    152 [768, 768] x+y
    153 [768, 768] x+y
    154 [768, 768] norm(x)
    155 [768, 768] x*y
    156 [768, 768] x+y
    157 [3072, 768] X*Y
    158 [3072, 768] x+y
    159 [3072, 768] unary(x)
    160 [768, 768] X*Y
    161 [768, 768] x+y
    162 [768, 768] x+y
    163 [768, 768] norm(x)
    164 [768, 768] x*y
    165 [768, 768] x+y
    166 [2304, 768] X*Y
    167 [2304, 768] x+y
    168 [768, 768] view(x)
    169 [589824, 1] view(x)
    170 [589824, 1] x-\>y
    171 [768, 768] view(x)
    172 [589824, 1] view(x)
    173 [589824, 1] x-\>y
    174 [589824, 1] view(x)
    175 [64, 12, 768] reshape(x)
    176 [768, 64, 12] permute(x)
    177 [768, 64, 12] cont(x)
    178 [589824, 1] view(x)
    179 [64, 12, 768] reshape(x)
    180 [64, 768, 12] permute(x)
    181 [768, 768] view(x)
    182 [64, 12, 768] cont(x)
    183 [64, 768, 12] permute(x)
    184 [768, 768, 12] X*Y
    185 [768, 768, 12] x*v
    186 [768, 768, 12] diag_mask_inf(x)
    187 [768, 768, 12] soft_max(x)
    188 [64, 768, 12] X*Y
    189 [64, 12, 768] permute(x)
    190 [768, 768] cont(x)
    191 [768, 768] X*Y
    192 [768, 768] x+y
    193 [768, 768] x+y
    194 [768, 768] norm(x)
    195 [768, 768] x*y
    196 [768, 768] x+y
    197 [3072, 768] X*Y
    198 [3072, 768] x+y
    199 [3072, 768] unary(x)
    200 [768, 768] X*Y
    201 [768, 768] x+y
    202 [768, 768] x+y
    203 [768, 768] norm(x)
    204 [768, 768] x*y
    205 [768, 768] x+y
    206 [2304, 768] X*Y
    207 [2304, 768] x+y
    208 [768, 768] view(x)
    209 [589824, 1] view(x)
    210 [589824, 1] x-\>y
    211 [768, 768] view(x)
    212 [589824, 1] view(x)
    213 [589824, 1] x-\>y
    214 [589824, 1] view(x)
    215 [64, 12, 768] reshape(x)
    216 [768, 64, 12] permute(x)
    217 [768, 64, 12] cont(x)
    218 [589824, 1] view(x)
    219 [64, 12, 768] reshape(x)
    220 [64, 768, 12] permute(x)
    221 [768, 768] view(x)
    222 [64, 12, 768] cont(x)
    223 [64, 768, 12] permute(x)
    224 [768, 768, 12] X*Y
    225 [768, 768, 12] x*v
    226 [768, 768, 12] diag_mask_inf(x)
    227 [768, 768, 12] soft_max(x)
    228 [64, 768, 12] X*Y
    229 [64, 12, 768] permute(x)
    230 [768, 768] cont(x)
    231 [768, 768] X*Y
    232 [768, 768] x+y
    233 [768, 768] x+y
    234 [768, 768] norm(x)
    235 [768, 768] x*y
    236 [768, 768] x+y
    237 [3072, 768] X*Y
    238 [3072, 768] x+y
    239 [3072, 768] unary(x)
    240 [768, 768] X*Y
    241 [768, 768] x+y
    242 [768, 768] x+y
    243 [768, 768] norm(x)
    244 [768, 768] x*y
    245 [768, 768] x+y
    246 [2304, 768] X*Y
    247 [2304, 768] x+y
    248 [768, 768] view(x)
    249 [589824, 1] view(x)
    250 [589824, 1] x-\>y
    251 [768, 768] view(x)
    252 [589824, 1] view(x)
    253 [589824, 1] x-\>y
    254 [589824, 1] view(x)
    255 [64, 12, 768] reshape(x)
    256 [768, 64, 12] permute(x)
    257 [768, 64, 12] cont(x)
    258 [589824, 1] view(x)
    259 [64, 12, 768] reshape(x)
    260 [64, 768, 12] permute(x)
    261 [768, 768] view(x)
    262 [64, 12, 768] cont(x)
    263 [64, 768, 12] permute(x)
    264 [768, 768, 12] X*Y
    265 [768, 768, 12] x*v
    266 [768, 768, 12] diag_mask_inf(x)
    267 [768, 768, 12] soft_max(x)
    268 [64, 768, 12] X*Y
    269 [64, 12, 768] permute(x)
    270 [768, 768] cont(x)
    271 [768, 768] X*Y
    272 [768, 768] x+y
    273 [768, 768] x+y
    274 [768, 768] norm(x)
    275 [768, 768] x*y
    276 [768, 768] x+y
    277 [3072, 768] X*Y
    278 [3072, 768] x+y
    279 [3072, 768] unary(x)
    280 [768, 768] X*Y
    281 [768, 768] x+y
    282 [768, 768] x+y
    283 [768, 768] norm(x)
    284 [768, 768] x*y
    285 [768, 768] x+y
    286 [2304, 768] X*Y
    287 [2304, 768] x+y
    288 [768, 768] view(x)
    289 [589824, 1] view(x)
    290 [589824, 1] x-\>y
    291 [768, 768] view(x)
    292 [589824, 1] view(x)
    293 [589824, 1] x-\>y
    294 [589824, 1] view(x)
    295 [64, 12, 768] reshape(x)
    296 [768, 64, 12] permute(x)
    297 [768, 64, 12] cont(x)
    298 [589824, 1] view(x)
    299 [64, 12, 768] reshape(x)
    300 [64, 768, 12] permute(x)
    301 [768, 768] view(x)
    302 [64, 12, 768] cont(x)
    303 [64, 768, 12] permute(x)
    304 [768, 768, 12] X*Y
    305 [768, 768, 12] x*v
    306 [768, 768, 12] diag_mask_inf(x)
    307 [768, 768, 12] soft_max(x)
    308 [64, 768, 12] X*Y
    309 [64, 12, 768] permute(x)
    310 [768, 768] cont(x)
    311 [768, 768] X*Y
    312 [768, 768] x+y
    313 [768, 768] x+y
    314 [768, 768] norm(x)
    315 [768, 768] x*y
    316 [768, 768] x+y
    317 [3072, 768] X*Y
    318 [3072, 768] x+y
    319 [3072, 768] unary(x)
    320 [768, 768] X*Y
    321 [768, 768] x+y
    322 [768, 768] x+y
    323 [768, 768] norm(x)
    324 [768, 768] x*y
    325 [768, 768] x+y
    326 [2304, 768] X*Y
    327 [2304, 768] x+y
    328 [768, 768] view(x)
    329 [589824, 1] view(x)
    330 [589824, 1] x-\>y
    331 [768, 768] view(x)
    332 [589824, 1] view(x)
    333 [589824, 1] x-\>y
    334 [589824, 1] view(x)
    335 [64, 12, 768] reshape(x)
    336 [768, 64, 12] permute(x)
    337 [768, 64, 12] cont(x)
    338 [589824, 1] view(x)
    339 [64, 12, 768] reshape(x)
    340 [64, 768, 12] permute(x)
    341 [768, 768] view(x)
    342 [64, 12, 768] cont(x)
    343 [64, 768, 12] permute(x)
    344 [768, 768, 12] X*Y
    345 [768, 768, 12] x*v
    346 [768, 768, 12] diag_mask_inf(x)
    347 [768, 768, 12] soft_max(x)
    348 [64, 768, 12] X*Y
    349 [64, 12, 768] permute(x)
    350 [768, 768] cont(x)
    351 [768, 768] X*Y
    352 [768, 768] x+y
    353 [768, 768] x+y
    354 [768, 768] norm(x)
    355 [768, 768] x*y
    356 [768, 768] x+y
    357 [3072, 768] X*Y
    358 [3072, 768] x+y
    359 [3072, 768] unary(x)
    360 [768, 768] X*Y
    361 [768, 768] x+y
    362 [768, 768] x+y
    363 [768, 768] norm(x)
    364 [768, 768] x*y
    365 [768, 768] x+y
    366 [2304, 768] X*Y
    367 [2304, 768] x+y
    368 [768, 768] view(x)
    369 [589824, 1] view(x)
    370 [589824, 1] x-\>y
    371 [768, 768] view(x)
    372 [589824, 1] view(x)
    373 [589824, 1] x-\>y
    374 [589824, 1] view(x)
    375 [64, 12, 768] reshape(x)
    376 [768, 64, 12] permute(x)
    377 [768, 64, 12] cont(x)
    378 [589824, 1] view(x)
    379 [64, 12, 768] reshape(x)
    380 [64, 768, 12] permute(x)
    381 [768, 768] view(x)
    382 [64, 12, 768] cont(x)
    383 [64, 768, 12] permute(x)
    384 [768, 768, 12] X*Y
    385 [768, 768, 12] x*v
    386 [768, 768, 12] diag_mask_inf(x)
    387 [768, 768, 12] soft_max(x)
    388 [64, 768, 12] X*Y
    389 [64, 12, 768] permute(x)
    390 [768, 768] cont(x)
    391 [768, 768] X*Y
    392 [768, 768] x+y
    393 [768, 768] x+y
    394 [768, 768] norm(x)
    395 [768, 768] x*y
    396 [768, 768] x+y
    397 [3072, 768] X*Y
    398 [3072, 768] x+y
    399 [3072, 768] unary(x)
    400 [768, 768] X*Y
    401 [768, 768] x+y
    402 [768, 768] x+y
    403 [768, 768] norm(x)
    404 [768, 768] x*y
    405 [768, 768] x+y
    406 [2304, 768] X*Y
    407 [2304, 768] x+y
    408 [768, 768] view(x)
    409 [589824, 1] view(x)
    410 [589824, 1] x-\>y
    411 [768, 768] view(x)
    412 [589824, 1] view(x)
    413 [589824, 1] x-\>y
    414 [589824, 1] view(x)
    415 [64, 12, 768] reshape(x)
    416 [768, 64, 12] permute(x)
    417 [768, 64, 12] cont(x)
    418 [589824, 1] view(x)
    419 [64, 12, 768] reshape(x)
    420 [64, 768, 12] permute(x)
    421 [768, 768] view(x)
    422 [64, 12, 768] cont(x)
    423 [64, 768, 12] permute(x)
    424 [768, 768, 12] X*Y
    425 [768, 768, 12] x*v
    426 [768, 768, 12] diag_mask_inf(x)
    427 [768, 768, 12] soft_max(x)
    428 [64, 768, 12] X*Y
    429 [64, 12, 768] permute(x)
    430 [768, 768] cont(x)
    431 [768, 768] X*Y
    432 [768, 768] x+y
    433 [768, 768] x+y
    434 [768, 768] norm(x)
    435 [768, 768] x*y
    436 [768, 768] x+y
    437 [3072, 768] X*Y
    438 [3072, 768] x+y
    439 [3072, 768] unary(x)
    440 [768, 768] X*Y
    441 [768, 768] x+y
    442 [768, 768] x+y
    443 [768, 768] norm(x)
    444 [768, 768] x*y
    445 [768, 768] x+y
    446 [2304, 768] X*Y
    447 [2304, 768] x+y
    448 [768, 768] view(x)
    449 [589824, 1] view(x)
    450 [589824, 1] x-\>y
    451 [768, 768] view(x)
    452 [589824, 1] view(x)
    453 [589824, 1] x-\>y
    454 [589824, 1] view(x)
    455 [64, 12, 768] reshape(x)
    456 [768, 64, 12] permute(x)
    457 [768, 64, 12] cont(x)
    458 [589824, 1] view(x)
    459 [64, 12, 768] reshape(x)
    460 [64, 768, 12] permute(x)
    461 [768, 768] view(x)
    462 [64, 12, 768] cont(x)
    463 [64, 768, 12] permute(x)
    464 [768, 768, 12] X*Y
    465 [768, 768, 12] x*v
    466 [768, 768, 12] diag_mask_inf(x)
    467 [768, 768, 12] soft_max(x)
    468 [64, 768, 12] X*Y
    469 [64, 12, 768] permute(x)
    470 [768, 768] cont(x)
    471 [768, 768] X*Y
    472 [768, 768] x+y
    473 [768, 768] x+y
    474 [768, 768] norm(x)
    475 [768, 768] x*y
    476 [768, 768] x+y
    477 [3072, 768] X*Y
    478 [3072, 768] x+y
    479 [3072, 768] unary(x)
    480 [768, 768] X*Y
    481 [768, 768] x+y
    482 [768, 768] x+y
    483 [768, 768] norm(x)
    484 [768, 768] x*y
    485 [768, 768] x+y
    486 [50257, 768] X*Y |}];

  Functions.model_uninit (addr model);
  keep model;
  ()
