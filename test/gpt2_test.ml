open Ctypes
open Gpt_2.C
open Model_explorer

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let attr key value = KeyValue.create ~key ~value
let pp_int64 fmt t = Format.fprintf fmt "%Ld" t
let pp_list p fmt t = Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@ ") p) t)
let pp_pair p1 p2 fmt (t1, t2) = Format.fprintf fmt "@[%a,@ %a@]" p1 t1 p2 t2

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

  type kind = Input | Output | Constant | Intermediate

  let kind_to_string kind =
    match kind with Input -> "Input" | Output -> "Output" | Constant -> "Constant" | Intermediate -> "Intermediate"

  type t = { id : int; kind : kind }
  type nodes = { map : t PtrMap.t; node_count : int; next : int }

  let empty node_count = { map = PtrMap.empty; node_count; next = node_count }
  let pp_addr fmt t = Format.fprintf fmt "%#LX" @@ Int64.of_nativeint t
  let pp fmt t = Format.fprintf fmt "@[{id:%d;@ kind:%s}" t.id @@ kind_to_string t.kind

  let add_node id tensor nodes =
    assert (id < nodes.node_count);
    let open Ggml.C in
    let t =
      let flags = getfp tensor Types.Tensor.flags in
      let kind =
        if Int32.logand flags Ggml_const.C.Types.tensor_flag_output = Int32.zero then Intermediate else Output
      in
      { id; kind }
    in
    let nodes =
      let ptr = Ctypes.raw_address_of_ptr @@ to_voidp tensor in
      { nodes with map = PtrMap.add ptr t nodes.map }
    in
    let src = getfp tensor Types.Tensor.src in
    let () =
      let l = CArray.to_list src |> List.filter (fun t -> not @@ is_null t) in
      if false then Format.eprintf "%d:not-null:%d@." id @@ List.length l;
      ()
    in
    CArray.fold_left
      (fun nodes tensor ->
        if is_null tensor then nodes
        else
          let ptr = Ctypes.raw_address_of_ptr @@ to_voidp tensor in
          if PtrMap.mem ptr nodes.map then
            let _ = if false then Format.eprintf "%d: duplicate ptr:%a@." id pp_addr ptr in
            nodes
          else
            let _ =
              if false then
                Format.eprintf "%d:added :%d %s@." id nodes.next
                @@ Ggml.C.Functions.op_name @@ getfp tensor Ggml.C.Types.Tensor.op
            in
            let id = nodes.next in
            let flags = getfp tensor Types.Tensor.flags in
            let kind =
              if Int32.logand flags Ggml_const.C.Types.tensor_flag_param <> Int32.zero then Constant
              else if Int32.logand flags Ggml_const.C.Types.tensor_flag_input <> Int32.zero then Input
              else if getfp tensor Ggml.C.Types.Tensor.op = Ggml.Types.Op.None then Constant
              else Intermediate
            in
            let t = { id; kind } in
            let map = PtrMap.add ptr t nodes.map in
            { nodes with map; next = succ nodes.next })
      nodes src

  let pp_nodes fmt t =
    let nodes = PtrMap.bindings t.map in
    if false then Format.(pp_print_list ~pp_sep:pp_print_newline (pp_pair pp_addr pp)) fmt nodes
    else Format.(pp_print_list ~pp_sep:pp_print_newline pp) fmt @@ List.map snd nodes
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
    {id:637; kind:Constant}
    {id:638; kind:Constant}
    {id:487; kind:Constant}
    {id:489; kind:Constant}
    {id:491; kind:Constant}
    {id:492; kind:Constant}
    {id:499; kind:Constant}
    {id:500; kind:Constant}
    {id:493; kind:Constant}
    {id:494; kind:Constant}
    {id:497; kind:Constant}
    {id:498; kind:Constant}
    {id:501; kind:Constant}
    {id:502; kind:Constant}
    {id:503; kind:Constant}
    {id:504; kind:Constant}
    {id:505; kind:Constant}
    {id:506; kind:Constant}
    {id:511; kind:Constant}
    {id:512; kind:Constant}
    {id:507; kind:Constant}
    {id:508; kind:Constant}
    {id:509; kind:Constant}
    {id:510; kind:Constant}
    {id:513; kind:Constant}
    {id:514; kind:Constant}
    {id:515; kind:Constant}
    {id:516; kind:Constant}
    {id:517; kind:Constant}
    {id:518; kind:Constant}
    {id:523; kind:Constant}
    {id:524; kind:Constant}
    {id:519; kind:Constant}
    {id:520; kind:Constant}
    {id:521; kind:Constant}
    {id:522; kind:Constant}
    {id:525; kind:Constant}
    {id:526; kind:Constant}
    {id:527; kind:Constant}
    {id:528; kind:Constant}
    {id:529; kind:Constant}
    {id:530; kind:Constant}
    {id:535; kind:Constant}
    {id:536; kind:Constant}
    {id:531; kind:Constant}
    {id:532; kind:Constant}
    {id:533; kind:Constant}
    {id:534; kind:Constant}
    {id:537; kind:Constant}
    {id:538; kind:Constant}
    {id:539; kind:Constant}
    {id:540; kind:Constant}
    {id:541; kind:Constant}
    {id:542; kind:Constant}
    {id:547; kind:Constant}
    {id:548; kind:Constant}
    {id:543; kind:Constant}
    {id:544; kind:Constant}
    {id:545; kind:Constant}
    {id:546; kind:Constant}
    {id:549; kind:Constant}
    {id:550; kind:Constant}
    {id:551; kind:Constant}
    {id:552; kind:Constant}
    {id:553; kind:Constant}
    {id:554; kind:Constant}
    {id:559; kind:Constant}
    {id:560; kind:Constant}
    {id:555; kind:Constant}
    {id:556; kind:Constant}
    {id:557; kind:Constant}
    {id:558; kind:Constant}
    {id:561; kind:Constant}
    {id:562; kind:Constant}
    {id:563; kind:Constant}
    {id:564; kind:Constant}
    {id:565; kind:Constant}
    {id:566; kind:Constant}
    {id:571; kind:Constant}
    {id:572; kind:Constant}
    {id:567; kind:Constant}
    {id:568; kind:Constant}
    {id:569; kind:Constant}
    {id:570; kind:Constant}
    {id:573; kind:Constant}
    {id:574; kind:Constant}
    {id:575; kind:Constant}
    {id:576; kind:Constant}
    {id:577; kind:Constant}
    {id:578; kind:Constant}
    {id:583; kind:Constant}
    {id:584; kind:Constant}
    {id:579; kind:Constant}
    {id:580; kind:Constant}
    {id:581; kind:Constant}
    {id:582; kind:Constant}
    {id:585; kind:Constant}
    {id:586; kind:Constant}
    {id:587; kind:Constant}
    {id:588; kind:Constant}
    {id:589; kind:Constant}
    {id:590; kind:Constant}
    {id:595; kind:Constant}
    {id:596; kind:Constant}
    {id:591; kind:Constant}
    {id:592; kind:Constant}
    {id:593; kind:Constant}
    {id:594; kind:Constant}
    {id:597; kind:Constant}
    {id:598; kind:Constant}
    {id:599; kind:Constant}
    {id:600; kind:Constant}
    {id:601; kind:Constant}
    {id:602; kind:Constant}
    {id:607; kind:Constant}
    {id:608; kind:Constant}
    {id:603; kind:Constant}
    {id:604; kind:Constant}
    {id:605; kind:Constant}
    {id:606; kind:Constant}
    {id:609; kind:Constant}
    {id:610; kind:Constant}
    {id:611; kind:Constant}
    {id:612; kind:Constant}
    {id:613; kind:Constant}
    {id:614; kind:Constant}
    {id:619; kind:Constant}
    {id:620; kind:Constant}
    {id:615; kind:Constant}
    {id:616; kind:Constant}
    {id:617; kind:Constant}
    {id:618; kind:Constant}
    {id:621; kind:Constant}
    {id:622; kind:Constant}
    {id:623; kind:Constant}
    {id:624; kind:Constant}
    {id:625; kind:Constant}
    {id:626; kind:Constant}
    {id:631; kind:Constant}
    {id:632; kind:Constant}
    {id:627; kind:Constant}
    {id:628; kind:Constant}
    {id:629; kind:Constant}
    {id:630; kind:Constant}
    {id:633; kind:Constant}
    {id:634; kind:Constant}
    {id:635; kind:Constant}
    {id:636; kind:Constant}
    {id:495; kind:Constant}
    {id:496; kind:Constant}
    {id:488; kind:Input}
    {id:490; kind:Input}
    {id:1; kind:Intermediate}
    {id:0; kind:Intermediate}
    {id:2; kind:Intermediate}
    {id:3; kind:Intermediate}
    {id:4; kind:Intermediate}
    {id:5; kind:Intermediate}
    {id:6; kind:Intermediate}
    {id:7; kind:Intermediate}
    {id:21; kind:Intermediate}
    {id:8; kind:Intermediate}
    {id:11; kind:Intermediate}
    {id:9; kind:Intermediate}
    {id:12; kind:Intermediate}
    {id:10; kind:Intermediate}
    {id:13; kind:Intermediate}
    {id:22; kind:Intermediate}
    {id:23; kind:Intermediate}
    {id:18; kind:Intermediate}
    {id:19; kind:Intermediate}
    {id:20; kind:Intermediate}
    {id:24; kind:Intermediate}
    {id:25; kind:Intermediate}
    {id:26; kind:Intermediate}
    {id:27; kind:Intermediate}
    {id:14; kind:Intermediate}
    {id:15; kind:Intermediate}
    {id:16; kind:Intermediate}
    {id:17; kind:Intermediate}
    {id:28; kind:Intermediate}
    {id:29; kind:Intermediate}
    {id:30; kind:Intermediate}
    {id:31; kind:Intermediate}
    {id:32; kind:Intermediate}
    {id:33; kind:Intermediate}
    {id:34; kind:Intermediate}
    {id:35; kind:Intermediate}
    {id:36; kind:Intermediate}
    {id:37; kind:Intermediate}
    {id:38; kind:Intermediate}
    {id:39; kind:Intermediate}
    {id:40; kind:Intermediate}
    {id:41; kind:Intermediate}
    {id:42; kind:Intermediate}
    {id:43; kind:Intermediate}
    {id:44; kind:Intermediate}
    {id:45; kind:Intermediate}
    {id:46; kind:Intermediate}
    {id:47; kind:Intermediate}
    {id:61; kind:Intermediate}
    {id:48; kind:Intermediate}
    {id:51; kind:Intermediate}
    {id:49; kind:Intermediate}
    {id:52; kind:Intermediate}
    {id:50; kind:Intermediate}
    {id:53; kind:Intermediate}
    {id:62; kind:Intermediate}
    {id:63; kind:Intermediate}
    {id:58; kind:Intermediate}
    {id:59; kind:Intermediate}
    {id:60; kind:Intermediate}
    {id:64; kind:Intermediate}
    {id:65; kind:Intermediate}
    {id:66; kind:Intermediate}
    {id:67; kind:Intermediate}
    {id:54; kind:Intermediate}
    {id:55; kind:Intermediate}
    {id:56; kind:Intermediate}
    {id:57; kind:Intermediate}
    {id:68; kind:Intermediate}
    {id:69; kind:Intermediate}
    {id:70; kind:Intermediate}
    {id:71; kind:Intermediate}
    {id:72; kind:Intermediate}
    {id:73; kind:Intermediate}
    {id:74; kind:Intermediate}
    {id:75; kind:Intermediate}
    {id:76; kind:Intermediate}
    {id:77; kind:Intermediate}
    {id:78; kind:Intermediate}
    {id:79; kind:Intermediate}
    {id:80; kind:Intermediate}
    {id:81; kind:Intermediate}
    {id:82; kind:Intermediate}
    {id:83; kind:Intermediate}
    {id:84; kind:Intermediate}
    {id:85; kind:Intermediate}
    {id:86; kind:Intermediate}
    {id:87; kind:Intermediate}
    {id:101; kind:Intermediate}
    {id:88; kind:Intermediate}
    {id:91; kind:Intermediate}
    {id:89; kind:Intermediate}
    {id:92; kind:Intermediate}
    {id:90; kind:Intermediate}
    {id:93; kind:Intermediate}
    {id:102; kind:Intermediate}
    {id:103; kind:Intermediate}
    {id:98; kind:Intermediate}
    {id:99; kind:Intermediate}
    {id:100; kind:Intermediate}
    {id:104; kind:Intermediate}
    {id:105; kind:Intermediate}
    {id:106; kind:Intermediate}
    {id:107; kind:Intermediate}
    {id:94; kind:Intermediate}
    {id:95; kind:Intermediate}
    {id:96; kind:Intermediate}
    {id:97; kind:Intermediate}
    {id:108; kind:Intermediate}
    {id:109; kind:Intermediate}
    {id:110; kind:Intermediate}
    {id:111; kind:Intermediate}
    {id:112; kind:Intermediate}
    {id:113; kind:Intermediate}
    {id:114; kind:Intermediate}
    {id:115; kind:Intermediate}
    {id:116; kind:Intermediate}
    {id:117; kind:Intermediate}
    {id:118; kind:Intermediate}
    {id:119; kind:Intermediate}
    {id:120; kind:Intermediate}
    {id:121; kind:Intermediate}
    {id:122; kind:Intermediate}
    {id:123; kind:Intermediate}
    {id:124; kind:Intermediate}
    {id:125; kind:Intermediate}
    {id:126; kind:Intermediate}
    {id:127; kind:Intermediate}
    {id:141; kind:Intermediate}
    {id:128; kind:Intermediate}
    {id:131; kind:Intermediate}
    {id:129; kind:Intermediate}
    {id:132; kind:Intermediate}
    {id:130; kind:Intermediate}
    {id:133; kind:Intermediate}
    {id:142; kind:Intermediate}
    {id:143; kind:Intermediate}
    {id:138; kind:Intermediate}
    {id:139; kind:Intermediate}
    {id:140; kind:Intermediate}
    {id:144; kind:Intermediate}
    {id:145; kind:Intermediate}
    {id:146; kind:Intermediate}
    {id:147; kind:Intermediate}
    {id:134; kind:Intermediate}
    {id:135; kind:Intermediate}
    {id:136; kind:Intermediate}
    {id:137; kind:Intermediate}
    {id:148; kind:Intermediate}
    {id:149; kind:Intermediate}
    {id:150; kind:Intermediate}
    {id:151; kind:Intermediate}
    {id:152; kind:Intermediate}
    {id:153; kind:Intermediate}
    {id:154; kind:Intermediate}
    {id:155; kind:Intermediate}
    {id:156; kind:Intermediate}
    {id:157; kind:Intermediate}
    {id:158; kind:Intermediate}
    {id:159; kind:Intermediate}
    {id:160; kind:Intermediate}
    {id:161; kind:Intermediate}
    {id:162; kind:Intermediate}
    {id:163; kind:Intermediate}
    {id:164; kind:Intermediate}
    {id:165; kind:Intermediate}
    {id:166; kind:Intermediate}
    {id:167; kind:Intermediate}
    {id:181; kind:Intermediate}
    {id:168; kind:Intermediate}
    {id:171; kind:Intermediate}
    {id:169; kind:Intermediate}
    {id:172; kind:Intermediate}
    {id:170; kind:Intermediate}
    {id:173; kind:Intermediate}
    {id:182; kind:Intermediate}
    {id:183; kind:Intermediate}
    {id:178; kind:Intermediate}
    {id:179; kind:Intermediate}
    {id:180; kind:Intermediate}
    {id:184; kind:Intermediate}
    {id:185; kind:Intermediate}
    {id:186; kind:Intermediate}
    {id:187; kind:Intermediate}
    {id:174; kind:Intermediate}
    {id:175; kind:Intermediate}
    {id:176; kind:Intermediate}
    {id:177; kind:Intermediate}
    {id:188; kind:Intermediate}
    {id:189; kind:Intermediate}
    {id:190; kind:Intermediate}
    {id:191; kind:Intermediate}
    {id:192; kind:Intermediate}
    {id:193; kind:Intermediate}
    {id:194; kind:Intermediate}
    {id:195; kind:Intermediate}
    {id:196; kind:Intermediate}
    {id:197; kind:Intermediate}
    {id:198; kind:Intermediate}
    {id:199; kind:Intermediate}
    {id:200; kind:Intermediate}
    {id:201; kind:Intermediate}
    {id:202; kind:Intermediate}
    {id:203; kind:Intermediate}
    {id:204; kind:Intermediate}
    {id:205; kind:Intermediate}
    {id:206; kind:Intermediate}
    {id:207; kind:Intermediate}
    {id:221; kind:Intermediate}
    {id:208; kind:Intermediate}
    {id:211; kind:Intermediate}
    {id:209; kind:Intermediate}
    {id:212; kind:Intermediate}
    {id:210; kind:Intermediate}
    {id:213; kind:Intermediate}
    {id:222; kind:Intermediate}
    {id:223; kind:Intermediate}
    {id:218; kind:Intermediate}
    {id:219; kind:Intermediate}
    {id:220; kind:Intermediate}
    {id:224; kind:Intermediate}
    {id:225; kind:Intermediate}
    {id:226; kind:Intermediate}
    {id:227; kind:Intermediate}
    {id:214; kind:Intermediate}
    {id:215; kind:Intermediate}
    {id:216; kind:Intermediate}
    {id:217; kind:Intermediate}
    {id:228; kind:Intermediate}
    {id:229; kind:Intermediate}
    {id:230; kind:Intermediate}
    {id:231; kind:Intermediate}
    {id:232; kind:Intermediate}
    {id:233; kind:Intermediate}
    {id:234; kind:Intermediate}
    {id:235; kind:Intermediate}
    {id:236; kind:Intermediate}
    {id:237; kind:Intermediate}
    {id:238; kind:Intermediate}
    {id:239; kind:Intermediate}
    {id:240; kind:Intermediate}
    {id:241; kind:Intermediate}
    {id:242; kind:Intermediate}
    {id:243; kind:Intermediate}
    {id:244; kind:Intermediate}
    {id:245; kind:Intermediate}
    {id:246; kind:Intermediate}
    {id:247; kind:Intermediate}
    {id:261; kind:Intermediate}
    {id:248; kind:Intermediate}
    {id:251; kind:Intermediate}
    {id:249; kind:Intermediate}
    {id:252; kind:Intermediate}
    {id:250; kind:Intermediate}
    {id:253; kind:Intermediate}
    {id:262; kind:Intermediate}
    {id:263; kind:Intermediate}
    {id:258; kind:Intermediate}
    {id:259; kind:Intermediate}
    {id:260; kind:Intermediate}
    {id:264; kind:Intermediate}
    {id:265; kind:Intermediate}
    {id:266; kind:Intermediate}
    {id:267; kind:Intermediate}
    {id:254; kind:Intermediate}
    {id:255; kind:Intermediate}
    {id:256; kind:Intermediate}
    {id:257; kind:Intermediate}
    {id:268; kind:Intermediate}
    {id:269; kind:Intermediate}
    {id:270; kind:Intermediate}
    {id:271; kind:Intermediate}
    {id:272; kind:Intermediate}
    {id:273; kind:Intermediate}
    {id:274; kind:Intermediate}
    {id:275; kind:Intermediate}
    {id:276; kind:Intermediate}
    {id:277; kind:Intermediate}
    {id:278; kind:Intermediate}
    {id:279; kind:Intermediate}
    {id:280; kind:Intermediate}
    {id:281; kind:Intermediate}
    {id:282; kind:Intermediate}
    {id:283; kind:Intermediate}
    {id:284; kind:Intermediate}
    {id:285; kind:Intermediate}
    {id:286; kind:Intermediate}
    {id:287; kind:Intermediate}
    {id:301; kind:Intermediate}
    {id:288; kind:Intermediate}
    {id:291; kind:Intermediate}
    {id:289; kind:Intermediate}
    {id:292; kind:Intermediate}
    {id:290; kind:Intermediate}
    {id:293; kind:Intermediate}
    {id:302; kind:Intermediate}
    {id:303; kind:Intermediate}
    {id:298; kind:Intermediate}
    {id:299; kind:Intermediate}
    {id:300; kind:Intermediate}
    {id:304; kind:Intermediate}
    {id:305; kind:Intermediate}
    {id:306; kind:Intermediate}
    {id:307; kind:Intermediate}
    {id:294; kind:Intermediate}
    {id:295; kind:Intermediate}
    {id:296; kind:Intermediate}
    {id:297; kind:Intermediate}
    {id:308; kind:Intermediate}
    {id:309; kind:Intermediate}
    {id:310; kind:Intermediate}
    {id:311; kind:Intermediate}
    {id:312; kind:Intermediate}
    {id:313; kind:Intermediate}
    {id:314; kind:Intermediate}
    {id:315; kind:Intermediate}
    {id:316; kind:Intermediate}
    {id:317; kind:Intermediate}
    {id:318; kind:Intermediate}
    {id:319; kind:Intermediate}
    {id:320; kind:Intermediate}
    {id:321; kind:Intermediate}
    {id:322; kind:Intermediate}
    {id:323; kind:Intermediate}
    {id:324; kind:Intermediate}
    {id:325; kind:Intermediate}
    {id:326; kind:Intermediate}
    {id:327; kind:Intermediate}
    {id:341; kind:Intermediate}
    {id:328; kind:Intermediate}
    {id:331; kind:Intermediate}
    {id:329; kind:Intermediate}
    {id:332; kind:Intermediate}
    {id:330; kind:Intermediate}
    {id:333; kind:Intermediate}
    {id:342; kind:Intermediate}
    {id:343; kind:Intermediate}
    {id:338; kind:Intermediate}
    {id:339; kind:Intermediate}
    {id:340; kind:Intermediate}
    {id:344; kind:Intermediate}
    {id:345; kind:Intermediate}
    {id:346; kind:Intermediate}
    {id:347; kind:Intermediate}
    {id:334; kind:Intermediate}
    {id:335; kind:Intermediate}
    {id:336; kind:Intermediate}
    {id:337; kind:Intermediate}
    {id:348; kind:Intermediate}
    {id:349; kind:Intermediate}
    {id:350; kind:Intermediate}
    {id:351; kind:Intermediate}
    {id:352; kind:Intermediate}
    {id:353; kind:Intermediate}
    {id:354; kind:Intermediate}
    {id:355; kind:Intermediate}
    {id:356; kind:Intermediate}
    {id:357; kind:Intermediate}
    {id:358; kind:Intermediate}
    {id:359; kind:Intermediate}
    {id:360; kind:Intermediate}
    {id:361; kind:Intermediate}
    {id:362; kind:Intermediate}
    {id:363; kind:Intermediate}
    {id:364; kind:Intermediate}
    {id:365; kind:Intermediate}
    {id:366; kind:Intermediate}
    {id:367; kind:Intermediate}
    {id:381; kind:Intermediate}
    {id:368; kind:Intermediate}
    {id:371; kind:Intermediate}
    {id:369; kind:Intermediate}
    {id:372; kind:Intermediate}
    {id:370; kind:Intermediate}
    {id:373; kind:Intermediate}
    {id:382; kind:Intermediate}
    {id:383; kind:Intermediate}
    {id:378; kind:Intermediate}
    {id:379; kind:Intermediate}
    {id:380; kind:Intermediate}
    {id:384; kind:Intermediate}
    {id:385; kind:Intermediate}
    {id:386; kind:Intermediate}
    {id:387; kind:Intermediate}
    {id:374; kind:Intermediate}
    {id:375; kind:Intermediate}
    {id:376; kind:Intermediate}
    {id:377; kind:Intermediate}
    {id:388; kind:Intermediate}
    {id:389; kind:Intermediate}
    {id:390; kind:Intermediate}
    {id:391; kind:Intermediate}
    {id:392; kind:Intermediate}
    {id:393; kind:Intermediate}
    {id:394; kind:Intermediate}
    {id:395; kind:Intermediate}
    {id:396; kind:Intermediate}
    {id:397; kind:Intermediate}
    {id:398; kind:Intermediate}
    {id:399; kind:Intermediate}
    {id:400; kind:Intermediate}
    {id:401; kind:Intermediate}
    {id:402; kind:Intermediate}
    {id:403; kind:Intermediate}
    {id:404; kind:Intermediate}
    {id:405; kind:Intermediate}
    {id:406; kind:Intermediate}
    {id:407; kind:Intermediate}
    {id:421; kind:Intermediate}
    {id:408; kind:Intermediate}
    {id:411; kind:Intermediate}
    {id:409; kind:Intermediate}
    {id:412; kind:Intermediate}
    {id:410; kind:Intermediate}
    {id:413; kind:Intermediate}
    {id:422; kind:Intermediate}
    {id:423; kind:Intermediate}
    {id:418; kind:Intermediate}
    {id:419; kind:Intermediate}
    {id:420; kind:Intermediate}
    {id:424; kind:Intermediate}
    {id:425; kind:Intermediate}
    {id:426; kind:Intermediate}
    {id:427; kind:Intermediate}
    {id:414; kind:Intermediate}
    {id:415; kind:Intermediate}
    {id:416; kind:Intermediate}
    {id:417; kind:Intermediate}
    {id:428; kind:Intermediate}
    {id:429; kind:Intermediate}
    {id:430; kind:Intermediate}
    {id:431; kind:Intermediate}
    {id:432; kind:Intermediate}
    {id:433; kind:Intermediate}
    {id:434; kind:Intermediate}
    {id:435; kind:Intermediate}
    {id:436; kind:Intermediate}
    {id:437; kind:Intermediate}
    {id:438; kind:Intermediate}
    {id:439; kind:Intermediate}
    {id:440; kind:Intermediate}
    {id:441; kind:Intermediate}
    {id:442; kind:Intermediate}
    {id:443; kind:Intermediate}
    {id:444; kind:Intermediate}
    {id:445; kind:Intermediate}
    {id:446; kind:Intermediate}
    {id:447; kind:Intermediate}
    {id:461; kind:Intermediate}
    {id:448; kind:Intermediate}
    {id:451; kind:Intermediate}
    {id:449; kind:Intermediate}
    {id:452; kind:Intermediate}
    {id:450; kind:Intermediate}
    {id:453; kind:Intermediate}
    {id:462; kind:Intermediate}
    {id:463; kind:Intermediate}
    {id:458; kind:Intermediate}
    {id:459; kind:Intermediate}
    {id:460; kind:Intermediate}
    {id:464; kind:Intermediate}
    {id:465; kind:Intermediate}
    {id:466; kind:Intermediate}
    {id:467; kind:Intermediate}
    {id:454; kind:Intermediate}
    {id:455; kind:Intermediate}
    {id:456; kind:Intermediate}
    {id:457; kind:Intermediate}
    {id:468; kind:Intermediate}
    {id:469; kind:Intermediate}
    {id:470; kind:Intermediate}
    {id:471; kind:Intermediate}
    {id:472; kind:Intermediate}
    {id:473; kind:Intermediate}
    {id:474; kind:Intermediate}
    {id:475; kind:Intermediate}
    {id:476; kind:Intermediate}
    {id:477; kind:Intermediate}
    {id:478; kind:Intermediate}
    {id:479; kind:Intermediate}
    {id:480; kind:Intermediate}
    {id:481; kind:Intermediate}
    {id:482; kind:Intermediate}
    {id:483; kind:Intermediate}
    {id:484; kind:Intermediate}
    {id:485; kind:Intermediate}
    {id:486; kind:Output} |}];

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
