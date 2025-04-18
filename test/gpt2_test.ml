open Ctypes
open Gpt_2.C
open Model_explorer

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let attr key value = KeyValue.create ~key ~value

let tensor n t =
  let name = getfp t Ggml.C.Types.Tensor.name in
  let name = to_string @@ CArray.start name in
  let name =
    let printed_name = if String.length name == 0 || String.starts_with ~prefix:"leaf_" name || String.starts_with ~prefix:"node_" name then ""
    else name ^ " " in
    Printf.sprintf "%s(%s)" printed_name @@ Ggml.C.Functions.type_name @@ getfp t Ggml.C.Types.Tensor.typ_ in
  let tensor_name = attr "tensor_name" name in
  let tensor_index = attr "tensor_index" @@ string_of_int n in
  let tensor_shape =
    let ne = getfp t Ggml.C.Types.Tensor.ne in
    let dims =
    if Ggml.C.Functions.is_matrix t then Printf.sprintf "[%Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1)
    else Printf.sprintf "[%Ld, %Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1) (CArray.get ne 2) in
    Printf.sprintf "%d %s %s" n dims (Ggml.C.Functions.op_symbol @@ getfp t Ggml.C.Types.Tensor.op) in
  ignore (tensor_name,tensor_index);
  ignore tensor_shape;
  name


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
  [%expect {| nodes:487 |}];

  let nodes = Array.init nodes (fun n -> Ggml.C.Functions.graph_node gpt2 n) in
  let names = Array.map (fun t -> Ggml.C.Functions.op_name @@ getfp t Ggml.C.Types.Tensor.op) nodes in
  ignore names;
  let names = Array.mapi tensor nodes in

  Format.printf "@[%a@]" (Format.pp_print_list ~pp_sep:Format.pp_print_newline Format.pp_print_string)
  @@ Array.to_list names;
  [%expect
    {|
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (f32)
     (view) (copy of  (view)) (f32)
     (view) (f32)
     (view) (f32)
     (view) (copy of  (view)) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
     (view) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (view) (f32)
     (view) (cont) (f32)
     (view) (cont) (permuted) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
     (permuted) (f32)
     (permuted) (cont) (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    (f32)
    logits (f32) |}];

  Functions.model_uninit (addr model);
  keep model;
  ()
