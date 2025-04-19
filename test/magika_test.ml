open Ctypes
open Magika.C

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let print pp t = Format.printf "@[%a@]@.%!" pp t

let%expect_test "magika" =
  let model = make Types.model_t in
  let fname = "models/magika.h5.gguf" in
  let rc = Functions.model_init (addr model) fname in
  assert (rc = 0);
  let magika = Functions.model_graph @@ addr model in
  let nodes = Ggml.C.Functions.graph_n_nodes magika in
  Format.printf "nodes:%u" nodes;
  [%expect "nodes:27"];
  let open Ggml_model_explorer in
  print TensorId.pp_nodes @@ TensorId.of_graph magika;
  [%expect
    {|
      {id:0; kind:Intermediate}
      {id:1; kind:Intermediate}
      {id:2; kind:Intermediate}
      {id:3; kind:Intermediate}
      {id:4; kind:Intermediate}
      {id:5; kind:Intermediate}
      {id:6; kind:Intermediate}
      {id:7; kind:Intermediate}
      {id:8; kind:Intermediate}
      {id:9; kind:Intermediate}
      {id:10; kind:Intermediate}
      {id:11; kind:Intermediate}
      {id:12; kind:Intermediate}
      {id:13; kind:Intermediate}
      {id:14; kind:Intermediate}
      {id:15; kind:Intermediate}
      {id:16; kind:Intermediate}
      {id:17; kind:Intermediate}
      {id:18; kind:Intermediate}
      {id:19; kind:Intermediate}
      {id:20; kind:Intermediate}
      {id:21; kind:Intermediate}
      {id:22; kind:Intermediate}
      {id:23; kind:Intermediate}
      {id:24; kind:Intermediate}
      {id:25; kind:Intermediate}
      {id:26; kind:Output}
      {id:27; kind:Constant}
      {id:28; kind:Input}
      {id:29; kind:Constant}
      {id:30; kind:Constant}
      {id:31; kind:Constant}
      {id:32; kind:Constant}
      {id:33; kind:Constant}
      {id:34; kind:Constant}
      {id:35; kind:Constant}
      {id:36; kind:Constant}
      {id:37; kind:Constant}
      {id:38; kind:Constant}
      {id:39; kind:Constant}
    |}];
  Functions.model_uninit @@ addr model;
  keep model;
  ()
