open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F
  let ns name = "ggml_" ^ name
  let _NS name = "GGML_" ^ name

  let make_enum name values =
    let _NAME v = _NS @@ String.uppercase_ascii name ^ "_" ^ v in
    enum (ns name) @@ List.map (fun (t, name) -> (t, constant (_NAME name) int64_t)) values

  let status = make_enum "status" Types.Status.values
  let typ = make_enum "type" Types.Type.values
  let prec = make_enum "prec" Types.Prec.values
  let ftype = make_enum "ftype" Types.Ftype.values
  let op = make_enum "op" Types.Op.values
  let unary_op = make_enum "unary_op" Types.UnaryOp.values
  let object_type = make_enum "object_type" Types.ObjectType.values
  let log_level = make_enum "log_level" Types.LogLevel.values
  let tensor_flag = make_enum "tensor_flag" Types.TensorFlag.values
  let op_pool = make_enum "op_pool" Types.OpPool.values
  let sort_order = make_enum "sort_order" Types.SortOrder.values
  let object' : [ `Object ] structure typ = structure (ns "object")
  let context : [ `Context ] structure typ = structure (ns "context")
  let cgraph : [ `Cgraph ] structure typ = structure (ns "cgraph")
  let backend_buffer : [ `BackendBuffer ] structure typ = structure (ns "backend_buffer")

  (*
  module InitParams = struct
    type t

    let t : t structure typ = structure (ns "init_params")
    let mem_size = field t "mem_size" size_t
    let mem_buffer = field t "mem_buffer" @@ ptr void
    let no_alloc = field t "no_alloc" bool
    let () = seal t
  end

  module Tensor = struct
    type t
    let t : t structure typ = structure (ns "tensor")
    (* Need to declare t before using it recursively in src and view_src *)
    let typ_ = field t "type" typ
    let buffer = field t "buffer" (ptr backend_buffer)
    let ne = field t "ne" (array max_dims int64_t)
    let nb = field t "nb" (array max_dims size_t)
    let op_ = field t "op" op
    (* GGML_MAX_OP_PARAMS is 64, sizeof(int32_t) is 4, so array size is 16 *)
    let op_params = field t "op_params" (array 16 int32_t)
    let flags = field t "flags" int32_t
    let src = field t "src" (array max_src (ptr t))
    let view_src = field t "view_src" (ptr t)
    let view_offs = field t "view_offs" size_t
    let data = field t "data" (ptr void)
    let name = field t "name" (array max_name char)
    let extra = field t "extra" (ptr void)
    let padding = field t "padding" (array 8 char)
    let () = seal t
  end
  *)
end
