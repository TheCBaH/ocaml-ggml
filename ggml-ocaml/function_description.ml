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

  (* Graph Computation *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> tensor @-> returning void)
end
