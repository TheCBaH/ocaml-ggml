open Ctypes
module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types

  let ns name = "ggml_" ^ name

  (* Context *)
  let init = foreign (ns "init") (InitParams.t @-> returning context)
  let free = foreign (ns "free") (context @-> returning void)
  let used_mem = foreign (ns "used_mem") (context @-> returning size_t)

  (* Types / Ops Info *)
  let type_name = foreign (ns "type_name") (typ @-> returning string)
  let op_name = foreign (ns "op_name") (op @-> returning string)

  (* Tensor Info *)
  let element_size = foreign (ns "element_size") (tensor @-> returning size_t)
  let nelements = foreign (ns "nelements") (tensor @-> returning int64_t)
  let nbytes = foreign (ns "nbytes") (tensor @-> returning size_t)

  (* Tensor Creation *)
  let new_tensor = foreign (ns "new_tensor") (context @-> typ @-> int @-> ptr int64_t @-> returning tensor)
  let new_tensor_1d = foreign (ns "new_tensor_1d") (context @-> typ @-> int64_t @-> returning tensor)
  let new_tensor_2d = foreign (ns "new_tensor_2d") (context @-> typ @-> int64_t @-> int64_t @-> returning tensor)

  let new_tensor_3d =
    foreign (ns "new_tensor_3d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  let new_tensor_4d =
    foreign (ns "new_tensor_4d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (* Basic Tensor Ops *)
  let dup = foreign (ns "dup") (context @-> tensor @-> returning tensor)
  let add = foreign (ns "add") (context @-> tensor @-> tensor @-> returning tensor)
  let sub = foreign (ns "sub") (context @-> tensor @-> tensor @-> returning tensor)
  let mul = foreign (ns "mul") (context @-> tensor @-> tensor @-> returning tensor)
  let div = foreign (ns "div") (context @-> tensor @-> tensor @-> returning tensor)
  let sqr = foreign (ns "sqr") (context @-> tensor @-> returning tensor)
  let sqrt = foreign (ns "sqrt") (context @-> tensor @-> returning tensor)
  let log = foreign (ns "log") (context @-> tensor @-> returning tensor)
  let sum = foreign (ns "sum") (context @-> tensor @-> returning tensor)
  let mean = foreign (ns "mean") (context @-> tensor @-> returning tensor)
  let abs = foreign (ns "abs") (context @-> tensor @-> returning tensor)
  let neg = foreign (ns "neg") (context @-> tensor @-> returning tensor)

  (* Tensor Manipulation Ops *)
  let reshape = foreign (ns "reshape") (context @-> tensor @-> tensor @-> returning tensor)
  let reshape_1d = foreign (ns "reshape_1d") (context @-> tensor @-> int64_t @-> returning tensor)
  let reshape_2d = foreign (ns "reshape_2d") (context @-> tensor @-> int64_t @-> int64_t @-> returning tensor)

  let reshape_3d =
    foreign (ns "reshape_3d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  let reshape_4d =
    foreign (ns "reshape_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  let view_1d = foreign (ns "view_1d") (context @-> tensor @-> int64_t @-> size_t @-> returning tensor)

  let view_2d =
    foreign (ns "view_2d") (context @-> tensor @-> int64_t @-> int64_t @-> size_t @-> size_t @-> returning tensor)

  let view_3d =
    foreign (ns "view_3d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  let view_4d =
    foreign (ns "view_4d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> size_t
     @-> returning tensor)

  let permute = foreign (ns "permute") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)
  let transpose = foreign (ns "transpose") (context @-> tensor @-> returning tensor)
  let get_rows = foreign (ns "get_rows") (context @-> tensor @-> tensor @-> returning tensor)
  let cpy = foreign (ns "cpy") (context @-> tensor @-> tensor @-> returning tensor)
  let cont = foreign (ns "cont") (context @-> tensor @-> returning tensor)
  let cast = foreign (ns "cast") (context @-> tensor @-> typ @-> returning tensor)

  (* Buffer Creation *)
  let new_buffer = foreign (ns "new_buffer") (context @-> size_t @-> returning (ptr void))

  (* Tensor Duplication / Viewing *)
  let dup_tensor = foreign (ns "dup_tensor") (context @-> tensor @-> returning tensor)
  let view_tensor = foreign (ns "view_tensor") (context @-> tensor @-> returning tensor)

  (* Context Tensor Enumeration and Lookup *)
  let get_first_tensor = foreign (ns "get_first_tensor") (context @-> returning tensor)
  let get_next_tensor = foreign (ns "get_next_tensor") (context @-> tensor @-> returning tensor)
  let get_tensor = foreign (ns "get_tensor") (context @-> string @-> returning tensor)

  (* Indexing *)
  let unravel_index =
    foreign (ns "unravel_index")
      (tensor @-> int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> returning void)

  (* Op Info *)
  let get_unary_op = foreign (ns "get_unary_op") (tensor @-> returning unary_op)

  (* Data Access *)
  let get_data = foreign (ns "get_data") (tensor @-> returning (ptr void))
  let get_data_f32 = foreign (ns "get_data_f32") (tensor @-> returning (ptr float))

  (* Tensor Naming *)
  let get_name = foreign (ns "get_name") (tensor @-> returning string)
  let set_name = foreign (ns "set_name") (tensor @-> string @-> returning tensor)
  (* ggml_format_name is variadic, skipping *)

  (* Tensor Flags *)
  let set_input = foreign (ns "set_input") (tensor @-> returning void)
  let set_output = foreign (ns "set_output") (tensor @-> returning void)
  let set_param = foreign (ns "set_param") (context @-> tensor @-> returning void)
  let set_loss = foreign (ns "set_loss") (tensor @-> returning void)

  (* Graph Computation *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> tensor @-> returning void)
end
