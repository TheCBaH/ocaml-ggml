open Ctypes
open Ggml.C
open Ggml_model_explorer
open Model_explorer

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

(* Helper to manually add a tensor and its ID to the map *)
let add_manual_tensor_id tensor id kind map =
  let ptr = raw_address_of_ptr @@ to_voidp tensor in
  TensorId.PtrMap.add ptr { TensorId.id; kind } map

(* Helper to print JSON for expect tests *)
let print_json jsont t = print_endline @@ Result.get_ok @@ Jsont_bytesrw.encode_string ~format:Jsont.Indent jsont t

let%expect_test "node: No inputs" =
  with_context (fun ctx ->
      let t_no_input = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      (* Manually create the nodes map *)
      let nodes_map =
        TensorId.PtrMap.empty |> add_manual_tensor_id t_no_input 0 TensorId.Constant
        (* Assign ID 0 *)
      in
      let nodes_struct = { TensorId.map = nodes_map; node_count = 1; next = 1 } in

      (* Call the function under test *)
      let graph_node_data = node nodes_struct t_no_input in

      (* Print the result as JSON *)
      print_json GraphNode.jsont graph_node_data;
      [%expect
        {|
        {
          "id": "0",
          "label": "pseudo_const",
          "namespace": "",
          "outputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "0"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            }
          ]
        } |}];
      keep t_no_input)

let%expect_test "node: One input (relu)" =
  with_context (fun ctx ->
      let t_in = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      let t_relu = Ggml.C.Functions.relu ctx t_in in

      (* Manually create the nodes map *)
      let nodes_map =
        TensorId.PtrMap.empty
        |> add_manual_tensor_id t_relu 5 TensorId.Intermediate (* Assign ID 5 to relu output *)
        |> add_manual_tensor_id t_in 10 TensorId.Constant (* Assign ID 10 to input *)
      in
      let nodes_struct = { TensorId.map = nodes_map; node_count = 11; next = 11 } in

      (* Call the function under test *)
      let graph_node_data = node nodes_struct t_relu in

      (* Print the result as JSON *)
      print_json GraphNode.jsont graph_node_data;
      [%expect
        {|
        {
          "id": "5",
          "label": "RELU",
          "namespace": "",
          "incomingEdges": [
            {
              "sourceNodeId": "10",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "0"
            }
          ],
          "inputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "10"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            }
          ],
          "outputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "5"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            }
          ]
        } |}];
      keep (t_in, t_relu))

let%expect_test "node: Two inputs (add)" =
  with_context (fun ctx ->
      let t_in1 = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      let t_in2 = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 10L in
      let t_add = Ggml.C.Functions.add ctx t_in1 t_in2 in

      (* Manually create the nodes map *)
      let nodes_map =
        TensorId.PtrMap.empty
        |> add_manual_tensor_id t_add 20 TensorId.Intermediate (* Assign ID 20 to add output *)
        |> add_manual_tensor_id t_in1 21 TensorId.Constant (* Assign ID 21 to input 1 *)
        |> add_manual_tensor_id t_in2 22 TensorId.Constant (* Assign ID 22 to input 2 *)
      in
      (* node_count and next should be >= highest ID + 1 *)
      let nodes_struct = { TensorId.map = nodes_map; node_count = 23; next = 23 } in

      (* Call the function under test *)
      let graph_node_data = node nodes_struct t_add in

      (* Print the result as JSON *)
      print_json GraphNode.jsont graph_node_data;
      [%expect
        {|
        {
          "id": "20",
          "label": "ADD",
          "namespace": "",
          "incomingEdges": [
            {
              "sourceNodeId": "21",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "0"
            },
            {
              "sourceNodeId": "22",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "1"
            }
          ],
          "inputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "21"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            },
            {
              "id": "1",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "22"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            }
          ],
          "outputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_index",
                  "value": "20"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[10]"
                }
              ]
            }
          ]
        } |}];
      keep (t_in1, t_in2, t_add))

(* Optional: Test with a named tensor *)
let%expect_test "node: Named tensor" =
  with_context (fun ctx ->
      let t_named = Ggml.C.Functions.new_tensor_1d ctx Ggml.Types.Type.F32 5L in
      let _ = Ggml.C.Functions.set_name t_named "my_special_tensor" in

      let nodes_map = TensorId.PtrMap.empty |> add_manual_tensor_id t_named 30 TensorId.Constant in
      let nodes_struct = { TensorId.map = nodes_map; node_count = 31; next = 31 } in

      let graph_node_data = node nodes_struct t_named in

      print_json GraphNode.jsont graph_node_data;
      [%expect
        {|
        {
          "id": "30",
          "label": "pseudo_const",
          "namespace": "",
          "outputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_name",
                  "value": "my_special_tensor"
                },
                {
                  "key": "tensor_index",
                  "value": "30"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[5]"
                }
              ]
            }
          ]
        } |}];
      keep t_named)
