open Ctypes
open Magika.C

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let print pp t = Format.printf "@[%a@]@.%!" pp t

let with_model f =
  let model = make Types.model_t in
  let fname = "models/magika.h5.gguf" in
  let rc = Functions.model_init (addr model) fname in
  assert (rc = 0);
  let magika = Functions.model_graph @@ addr model in
  Fun.protect
    ~finally:(fun () ->
      Functions.model_uninit @@ addr model;
      keep model;
      ())
    (fun () -> f magika)

let%expect_test "magika" =
  with_model (fun magika ->
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
    |}]);
  ()

let print_json jsont t = print_endline @@ Result.get_ok @@ Jsont_bytesrw.encode_string ~format:Jsont.Indent jsont t

let%expect_test "tensors" =
  with_model (fun magika ->
      let open Ggml_model_explorer in
      let tensors = TensorId.of_graph magika in
      print pp_shape @@ Option.get @@ TensorId.get_tensor tensors 19;
      [%expect "[256]"];
      let attrs = List.rev @@ TensorId.fold (fun t id l -> tensor id t :: l) tensors [] in
      print_json (Jsont.list Model_explorer.KeyValueList.jsont) attrs;
      [%expect
        {|
        [
          [
            {
              "key": "tensor_name",
              "value": "dense/kernel:0"
            },
            {
              "key": "tensor_index",
              "value": "27"
            },
            {
              "key": "tensor_shape",
              "value": "f32[257,128]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "dense/bias:0"
            },
            {
              "key": "tensor_index",
              "value": "29"
            },
            {
              "key": "tensor_shape",
              "value": "f32[128]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "layer_normalization/gamma:0"
            },
            {
              "key": "tensor_index",
              "value": "30"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "layer_normalization/beta:0"
            },
            {
              "key": "tensor_index",
              "value": "31"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "dense_1/kernel:0"
            },
            {
              "key": "tensor_index",
              "value": "32"
            },
            {
              "key": "tensor_shape",
              "value": "f32[512,256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "dense_1/bias:0"
            },
            {
              "key": "tensor_index",
              "value": "33"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "dense_2/kernel:0"
            },
            {
              "key": "tensor_index",
              "value": "34"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "dense_2/bias:0"
            },
            {
              "key": "tensor_index",
              "value": "35"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "layer_normalization_1/gamma:0"
            },
            {
              "key": "tensor_index",
              "value": "36"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "layer_normalization_1/beta:0"
            },
            {
              "key": "tensor_index",
              "value": "37"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "target_label/kernel:0"
            },
            {
              "key": "tensor_index",
              "value": "38"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,113]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "target_label/bias:0"
            },
            {
              "key": "tensor_index",
              "value": "39"
            },
            {
              "key": "tensor_shape",
              "value": "f32[113]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "input"
            },
            {
              "key": "tensor_index",
              "value": "28"
            },
            {
              "key": "tensor_shape",
              "value": "f32[257,1536]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "0"
            },
            {
              "key": "tensor_shape",
              "value": "f32[128,1536]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "1"
            },
            {
              "key": "tensor_shape",
              "value": "f32[128,1536]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "2"
            },
            {
              "key": "tensor_shape",
              "value": "f32[128,1536]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (reshaped)"
            },
            {
              "key": "tensor_index",
              "value": "3"
            },
            {
              "key": "tensor_shape",
              "value": "f32[512,384]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (reshaped) (transposed)"
            },
            {
              "key": "tensor_index",
              "value": "4"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,512]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (reshaped) (transposed) (cont)"
            },
            {
              "key": "tensor_index",
              "value": "5"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,512]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "6"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,512]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "7"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,512]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "8"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,512]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (transposed)"
            },
            {
              "key": "tensor_index",
              "value": "9"
            },
            {
              "key": "tensor_shape",
              "value": "f32[512,384]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (transposed) (cont)"
            },
            {
              "key": "tensor_index",
              "value": "10"
            },
            {
              "key": "tensor_shape",
              "value": "f32[512,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "11"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "12"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "13"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "14"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "15"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "16"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256,384]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (transposed)"
            },
            {
              "key": "tensor_index",
              "value": "17"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (transposed) (cont)"
            },
            {
              "key": "tensor_index",
              "value": "18"
            },
            {
              "key": "tensor_shape",
              "value": "f32[384,256]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "19"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": " (reshaped)"
            },
            {
              "key": "tensor_index",
              "value": "20"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "21"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "22"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "23"
            },
            {
              "key": "tensor_shape",
              "value": "f32[256]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "24"
            },
            {
              "key": "tensor_shape",
              "value": "f32[113]"
            }
          ],
          [
            {
              "key": "tensor_index",
              "value": "25"
            },
            {
              "key": "tensor_shape",
              "value": "f32[113]"
            }
          ],
          [
            {
              "key": "tensor_name",
              "value": "target_label_probs"
            },
            {
              "key": "tensor_index",
              "value": "26"
            },
            {
              "key": "tensor_shape",
              "value": "f32[113]"
            }
          ]
        ] |}];
      let node_0 = Ggml.C.Functions.graph_node magika 0 in
      print_json Model_explorer.GraphNode.jsont @@ node tensors node_0;
      [%expect
        {|
        {
          "id": "0",
          "label": "MUL_MAT",
          "namespace": "",
          "incomingEdges": [
            {
              "sourceNodeId": "27",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "0"
            },
            {
              "sourceNodeId": "28",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "1"
            }
          ],
          "inputsMetadata": [
            {
              "id": "0",
              "attrs": [
                {
                  "key": "tensor_name",
                  "value": "dense/kernel:0"
                },
                {
                  "key": "tensor_index",
                  "value": "27"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[257,128]"
                }
              ]
            },
            {
              "id": "1",
              "attrs": [
                {
                  "key": "tensor_name",
                  "value": "input"
                },
                {
                  "key": "tensor_index",
                  "value": "28"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[257,1536]"
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
                  "value": "0"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[128,1536]"
                }
              ]
            }
          ]
        } |}];
      let node_1 = Ggml.C.Functions.graph_node magika 1 in
      print_json Model_explorer.GraphNode.jsont @@ node tensors node_1;
      [%expect
        {|
        {
          "id": "1",
          "label": "ADD",
          "namespace": "",
          "incomingEdges": [
            {
              "sourceNodeId": "0",
              "sourceNodeOutputId": "0",
              "targetNodeInputId": "0"
            },
            {
              "sourceNodeId": "29",
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
                  "value": "0"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[128,1536]"
                }
              ]
            },
            {
              "id": "1",
              "attrs": [
                {
                  "key": "tensor_name",
                  "value": "dense/bias:0"
                },
                {
                  "key": "tensor_index",
                  "value": "29"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[128]"
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
                  "value": "1"
                },
                {
                  "key": "tensor_shape",
                  "value": "f32[128,1536]"
                }
              ]
            }
          ]
        } |}];
      ())
