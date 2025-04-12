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
  let element_size = foreign (ns "element_size") (ptr Tensor.t @-> returning size_t)
  let nelements = foreign (ns "nelements") (ptr Tensor.t @-> returning int64_t)
  let nbytes = foreign (ns "nbytes") (ptr Tensor.t @-> returning size_t)

  (* Tensor Creation *)
  let new_tensor = foreign (ns "new_tensor") (context @-> typ @-> int @-> ptr int64_t @-> returning (ptr Tensor.t))
  let new_tensor_1d = foreign (ns "new_tensor_1d") (context @-> typ @-> int64_t @-> returning (ptr Tensor.t))
  let new_tensor_2d = foreign (ns "new_tensor_2d") (context @-> typ @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  let new_tensor_3d =
    foreign (ns "new_tensor_3d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  let new_tensor_4d =
    foreign (ns "new_tensor_4d")
      (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  (* Basic Tensor Ops *)
  let dup = foreign (ns "dup") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let add = foreign (ns "add") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let sub = foreign (ns "sub") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let mul = foreign (ns "mul") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let div = foreign (ns "div") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let sqr = foreign (ns "sqr") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let sqrt = foreign (ns "sqrt") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let log = foreign (ns "log") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let sum = foreign (ns "sum") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let mean = foreign (ns "mean") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let abs = foreign (ns "abs") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let neg = foreign (ns "neg") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))

  (* Tensor Manipulation Ops *)
  let reshape = foreign (ns "reshape") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let reshape_1d = foreign (ns "reshape_1d") (context @-> ptr Tensor.t @-> int64_t @-> returning (ptr Tensor.t))

  let reshape_2d =
    foreign (ns "reshape_2d") (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  let reshape_3d =
    foreign (ns "reshape_3d") (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  let reshape_4d =
    foreign (ns "reshape_4d")
      (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning (ptr Tensor.t))

  let view_1d = foreign (ns "view_1d") (context @-> ptr Tensor.t @-> int64_t @-> size_t @-> returning (ptr Tensor.t))

  let view_2d =
    foreign (ns "view_2d")
      (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> returning (ptr Tensor.t))

  let view_3d =
    foreign (ns "view_3d")
      (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t
      @-> returning (ptr Tensor.t))

  let view_4d =
    foreign (ns "view_4d")
      (context @-> ptr Tensor.t @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t
     @-> size_t
      @-> returning (ptr Tensor.t))

  let permute =
    foreign (ns "permute") (context @-> ptr Tensor.t @-> int @-> int @-> int @-> int @-> returning (ptr Tensor.t))

  let transpose = foreign (ns "transpose") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let get_rows = foreign (ns "get_rows") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let cpy = foreign (ns "cpy") (context @-> ptr Tensor.t @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let cont = foreign (ns "cont") (context @-> ptr Tensor.t @-> returning (ptr Tensor.t))
  let cast = foreign (ns "cast") (context @-> ptr Tensor.t @-> typ @-> returning (ptr Tensor.t))

  (* Graph Computation *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> ptr Tensor.t @-> returning void)
end
