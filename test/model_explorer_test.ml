open Ctypes
open Ggml.C
open Ggml_model_explorer (* Assuming node_inputs is in this module *)

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))

(* Helper to manage ggml_context lifecycle *)
let with_context f =
  let params = make Types.InitParams.t in
  setf params Types.InitParams.no_alloc false;
  setf params Types.InitParams.mem_size @@ Unsigned.Size_t.of_int @@ (1024 * 1024);
  setf params Types.InitParams.mem_buffer null;
  let ctx = Ggml.C.Functions.init params in
  assert (not (is_null ctx));
  Fun.protect
    ~finally:(fun () ->
      Ggml.C.Functions.free ctx;
      keep ctx)
    (fun () -> f ctx)

(* Helper to print the results for expect tests *)
let pp_input_results fmt results =
  Format.fprintf fmt "Input count: %d\n" (List.length results);
  List.iteri
    (fun i (idx, ptr) ->
      (* Basic check: print index and whether pointer is non-null *)
      Format.fprintf fmt "Result %d: Index=%d, Is_Null=%b\n" i idx (is_null ptr))
    results

let%expect_test "node_inputs: No inputs" =
  with_context (fun ctx ->
      (* Create a tensor with no explicit inputs *)
      let t_no_input = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      let inputs = node_inputs t_no_input in
      pp_input_results Format.std_formatter inputs;
      [%expect {| Input count: 0 |}])

let%expect_test "node_inputs: One input (relu)" =
  with_context (fun ctx ->
      let t_in = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      (* Create a tensor using a unary operation *)
      let t_relu = Ggml.C.Functions.relu ctx t_in in
      let inputs = node_inputs t_relu in
      pp_input_results Format.std_formatter inputs;
      [%expect {|
      Input count: 1
      Result 0: Index=0, Is_Null=false |}];
      keep (t_in, t_relu);
      ())

let%expect_test "node_inputs: Two inputs (add)" =
  with_context (fun ctx ->
      let t_in1 = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      let t_in2 = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      (* Create a tensor using a binary operation *)
      let t_add = Ggml.C.Functions.add ctx t_in1 t_in2 in
      let inputs = node_inputs t_add in
      pp_input_results Format.std_formatter inputs;
      [%expect {|
      Input count: 2
      Result 0: Index=0, Is_Null=false
      Result 1: Index=1, Is_Null=false |}];
      keep (t_in1, t_in2, t_add))
