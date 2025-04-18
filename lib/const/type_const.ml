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
  let tensor_size = constant "GGML_TENSOR_SIZE" int
  let tensor_flag_input = constant "GGML_TENSOR_FLAG_INPUT" int32_t
  let tensor_flag_output = constant "GGML_TENSOR_FLAG_OUTPUT" int32_t
  let tensor_flag_param = constant "GGML_TENSOR_FLAG_PARAM" int32_t
  let tensor_flag_loss = constant "GGML_TENSOR_FLAG_LOSS" int32_t
end
