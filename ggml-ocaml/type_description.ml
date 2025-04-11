open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let max_dims = constant "GGML_MAX_DIMS" int
  let file_magic = constant "GGML_FILE_MAGIC" int
  let file_version = constant "GGML_FILE_VERSION" int
  let qnt_version = constant "GGML_QNT_VERSION" int
  let qnt_version_factor = constant "GGML_QNT_VERSION_FACTOR" int
  let max_params = constant "GGML_MAX_PARAMS" int
  let max_src = constant "GGML_MAX_SRC" int
  let max_n_threads = constant "GGML_MAX_N_THREADS" int
  let max_op_params = constant "GGML_MAX_OP_PARAMS" int
  let max_name = constant "GGML_MAX_NAME" int
  let default_n_threads = constant "GGML_DEFAULT_N_THREADS" int
  let default_graph_size = constant "GGML_DEFAULT_GRAPH_SIZE" int
  let mem_align = constant "GGML_MEM_ALIGN" int
  let exit_success = constant "GGML_EXIT_SUCCESS" int
  let exit_aborted = constant "GGML_EXIT_ABORTED" int
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

  module InitParams = struct
    type t

    let t : t structure typ = structure (ns "init_params")
    let mem_size = field t "mem_size" size_t
    let mem_buffer = field t "mem_buffer" @@ ptr void
    let no_alloc = field t "no_alloc" bool
    let () = seal t
  end
end
