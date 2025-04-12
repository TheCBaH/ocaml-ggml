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

  (* Operations with backpropagation *)
  let dup = foreign (ns "dup") (context @-> tensor @-> returning tensor)
  let dup_inplace = foreign (ns "dup_inplace") (context @-> tensor @-> returning tensor)
  let add = foreign (ns "add") (context @-> tensor @-> tensor @-> returning tensor)
  let add_inplace = foreign (ns "add_inplace") (context @-> tensor @-> tensor @-> returning tensor)
  let add_cast = foreign (ns "add_cast") (context @-> tensor @-> tensor @-> typ @-> returning tensor)
  let add1 = foreign (ns "add1") (context @-> tensor @-> tensor @-> returning tensor)
  let add1_inplace = foreign (ns "add1_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  let acc =
    foreign (ns "acc") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  let acc_inplace =
    foreign (ns "acc_inplace")
      (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  let sub = foreign (ns "sub") (context @-> tensor @-> tensor @-> returning tensor)
  let sub_inplace = foreign (ns "sub_inplace") (context @-> tensor @-> tensor @-> returning tensor)
  let mul = foreign (ns "mul") (context @-> tensor @-> tensor @-> returning tensor)
  let mul_inplace = foreign (ns "mul_inplace") (context @-> tensor @-> tensor @-> returning tensor)
  let div = foreign (ns "div") (context @-> tensor @-> tensor @-> returning tensor)
  let div_inplace = foreign (ns "div_inplace") (context @-> tensor @-> tensor @-> returning tensor)
  let sqr = foreign (ns "sqr") (context @-> tensor @-> returning tensor)
  let sqr_inplace = foreign (ns "sqr_inplace") (context @-> tensor @-> returning tensor)
  let sqrt = foreign (ns "sqrt") (context @-> tensor @-> returning tensor)
  let sqrt_inplace = foreign (ns "sqrt_inplace") (context @-> tensor @-> returning tensor)
  let log = foreign (ns "log") (context @-> tensor @-> returning tensor)
  let log_inplace = foreign (ns "log_inplace") (context @-> tensor @-> returning tensor)
  let sin = foreign (ns "sin") (context @-> tensor @-> returning tensor)
  let sin_inplace = foreign (ns "sin_inplace") (context @-> tensor @-> returning tensor)
  let cos = foreign (ns "cos") (context @-> tensor @-> returning tensor)
  let cos_inplace = foreign (ns "cos_inplace") (context @-> tensor @-> returning tensor)
  let sum = foreign (ns "sum") (context @-> tensor @-> returning tensor)
  let sum_rows = foreign (ns "sum_rows") (context @-> tensor @-> returning tensor)
  let mean = foreign (ns "mean") (context @-> tensor @-> returning tensor)
  let argmax = foreign (ns "argmax") (context @-> tensor @-> returning tensor)
  let count_equal = foreign (ns "count_equal") (context @-> tensor @-> tensor @-> returning tensor)
  let repeat = foreign (ns "repeat") (context @-> tensor @-> tensor @-> returning tensor)
  let repeat_back = foreign (ns "repeat_back") (context @-> tensor @-> tensor @-> returning tensor)
  let concat = foreign (ns "concat") (context @-> tensor @-> tensor @-> int @-> returning tensor)
  let abs = foreign (ns "abs") (context @-> tensor @-> returning tensor)
  let abs_inplace = foreign (ns "abs_inplace") (context @-> tensor @-> returning tensor)
  let sgn = foreign (ns "sgn") (context @-> tensor @-> returning tensor)
  let sgn_inplace = foreign (ns "sgn_inplace") (context @-> tensor @-> returning tensor)
  let neg = foreign (ns "neg") (context @-> tensor @-> returning tensor)
  let neg_inplace = foreign (ns "neg_inplace") (context @-> tensor @-> returning tensor)
  let step = foreign (ns "step") (context @-> tensor @-> returning tensor)
  let step_inplace = foreign (ns "step_inplace") (context @-> tensor @-> returning tensor)
  let tanh = foreign (ns "tanh") (context @-> tensor @-> returning tensor)
  let tanh_inplace = foreign (ns "tanh_inplace") (context @-> tensor @-> returning tensor)
  let elu = foreign (ns "elu") (context @-> tensor @-> returning tensor)
  let elu_inplace = foreign (ns "elu_inplace") (context @-> tensor @-> returning tensor)
  let relu = foreign (ns "relu") (context @-> tensor @-> returning tensor)
  let leaky_relu = foreign (ns "leaky_relu") (context @-> tensor @-> float @-> bool @-> returning tensor)
  let relu_inplace = foreign (ns "relu_inplace") (context @-> tensor @-> returning tensor)
  let sigmoid = foreign (ns "sigmoid") (context @-> tensor @-> returning tensor)
  let sigmoid_inplace = foreign (ns "sigmoid_inplace") (context @-> tensor @-> returning tensor)
  let gelu = foreign (ns "gelu") (context @-> tensor @-> returning tensor)
  let gelu_inplace = foreign (ns "gelu_inplace") (context @-> tensor @-> returning tensor)
  let gelu_quick = foreign (ns "gelu_quick") (context @-> tensor @-> returning tensor)
  let gelu_quick_inplace = foreign (ns "gelu_quick_inplace") (context @-> tensor @-> returning tensor)
  let silu = foreign (ns "silu") (context @-> tensor @-> returning tensor)
  let silu_inplace = foreign (ns "silu_inplace") (context @-> tensor @-> returning tensor)
  let silu_back = foreign (ns "silu_back") (context @-> tensor @-> tensor @-> returning tensor)
  let hardswish = foreign (ns "hardswish") (context @-> tensor @-> returning tensor)
  let hardsigmoid = foreign (ns "hardsigmoid") (context @-> tensor @-> returning tensor)
  let exp = foreign (ns "exp") (context @-> tensor @-> returning tensor)
  let exp_inplace = foreign (ns "exp_inplace") (context @-> tensor @-> returning tensor)
  let norm = foreign (ns "norm") (context @-> tensor @-> float @-> returning tensor)
  let norm_inplace = foreign (ns "norm_inplace") (context @-> tensor @-> float @-> returning tensor)
  let rms_norm = foreign (ns "rms_norm") (context @-> tensor @-> float @-> returning tensor)
  let rms_norm_inplace = foreign (ns "rms_norm_inplace") (context @-> tensor @-> float @-> returning tensor)
  let group_norm = foreign (ns "group_norm") (context @-> tensor @-> int @-> float @-> returning tensor)
  let group_norm_inplace = foreign (ns "group_norm_inplace") (context @-> tensor @-> int @-> float @-> returning tensor)
  let l2_norm = foreign (ns "l2_norm") (context @-> tensor @-> float @-> returning tensor)
  let l2_norm_inplace = foreign (ns "l2_norm_inplace") (context @-> tensor @-> float @-> returning tensor)
  let rms_norm_back = foreign (ns "rms_norm_back") (context @-> tensor @-> tensor @-> float @-> returning tensor)
  let mul_mat = foreign (ns "mul_mat") (context @-> tensor @-> tensor @-> returning tensor)
  let mul_mat_set_prec = foreign (ns "mul_mat_set_prec") (tensor @-> prec @-> returning void)
  let mul_mat_id = foreign (ns "mul_mat_id") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (* Graph Computation *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> tensor @-> returning void)
end
