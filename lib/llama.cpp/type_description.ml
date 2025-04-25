open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "llama_" ^ name
  let _NS name = "LLAMA_" ^ name

  let make_enum ?(_NS = _NS) ?_NAME ?(ns = ns) name values =
    let _NAME = match _NAME with Some _NAME -> _NAME | None -> String.uppercase_ascii name in
    let _NAME v = _NS @@ _NAME ^ "_" ^ v in
    enum (ns name) @@ List.map (fun (t, name) -> (t, constant (_NAME name) int64_t)) values

  (* Enums from llama_types.ml *)
  let vocab_type = make_enum "vocab_type" Types.VocabType.values
  let vocab_pre_type = make_enum "vocab_pre_type" Types.VocabPreType.values
  let rope_type = make_enum "rope_type" Types.RopeType.values
  let token_type = make_enum "token_type" Types.TokenType.values
  let token_attr = make_enum "token_attr" Types.TokenAttr.values (* Note: Bit flags *)
  let ftype = make_enum "ftype" Types.Ftype.values
  let rope_scaling_type = make_enum "rope_scaling_type" Types.RopeScalingType.values
  let pooling_type = make_enum "pooling_type" Types.PoolingType.values
  let attention_type = make_enum "attention_type" Types.AttentionType.values
  let split_mode = make_enum "split_mode" Types.SplitMode.values

  let model_kv_override_type =
    make_enum ~_NAME:"KV_OVERRIDE_TYPE" "model_kv_override_type" Types.ModelKvOverrideType.values

  let ggml_type = make_enum ~ns:Ggml.C.Types.ns ~_NS:Ggml.C.Types._NS "type" Ggml.Types.Type.values

  (* Typedefs *)
  let pos = typedef int32_t (ns "pos")
  let token = typedef int32_t (ns "token")
  let seq_id = typedef int32_t (ns "seq_id")

  (* Opaque struct types *)
  let struct_vocab : [ `Vocab ] structure typ = structure (ns "vocab")
  let vocab = ptr struct_vocab
  let const_vocab = ptr @@ const struct_vocab
  let struct_model : [ `Model ] structure typ = structure (ns "model")
  let model = ptr struct_model
  let const_model = ptr @@ const struct_model
  let struct_context : [ `Context ] structure typ = structure (ns "context")
  let context = ptr struct_context
  let struct_kv_cache : [ `KvCache ] structure typ = structure (ns "kv_cache")
  let kv_cache = ptr struct_kv_cache
  let struct_adapter_lora : [ `AdapterLora ] structure typ = structure (ns "adapter_lora")
  let adapter_lora = ptr struct_adapter_lora

  (* Function pointer types *)
  let progress_callback = static_funptr (float @-> ptr void @-> returning bool)

  (** Token data structure. *)
  module TokenData = struct
    type t

    let t : t structure typ = structure (ns "token_data")
    let id = field t "id" token
    let logit = field t "logit" float
    let p = field t "p" float
    let () = seal t
  end

  (** Token data array structure. *)
  module TokenDataArray = struct
    type t

    let t : t structure typ = structure (ns "token_data_array")
    let data = field t "data" (ptr TokenData.t)
    let size = field t "size" size_t
    let selected = field t "selected" int64_t
    let sorted = field t "sorted" bool
    let () = seal t
  end

  (** Batch structure for decoding. *)
  module Batch = struct
    type t

    let t : t structure typ = structure (ns "batch")
    let n_tokens = field t "n_tokens" int32_t
    let token = field t "token" (ptr token)
    let embd = field t "embd" (ptr float)
    let pos = field t "pos" (ptr pos)
    let n_seq_id = field t "n_seq_id" (ptr int32_t)
    let seq_id = field t "seq_id" (ptr (ptr seq_id)) (* llama_seq_id ** *)
    let logits = field t "logits" (ptr int8_t)
    let () = seal t
  end

  (** Model key-value override structure. *)
  module ModelKvOverride = struct
    type t

    let t : t structure typ = structure (ns "model_kv_override")
    let tag = field t "tag" model_kv_override_type
    let key = field t "key" (array 128 char)

    (* Union handling: Define accessors or a variant type later if needed *)
    let val_i64 = field t "val_i64" int64_t
    let val_f64 = field t "val_f64" double
    let val_bool = field t "val_bool" bool
    let val_str = field t "val_str" (array 128 char)
    let () = seal t
  end

  (** Model tensor buffer type override structure. *)
  module ModelTensorBuftOverride = struct
    type t

    let t : t structure typ = structure (ns "model_tensor_buft_override")
    let pattern = field t "pattern" string (* const char * *)

    (* TODO: Define ggml_backend_buffer_type_t enum in ggml bindings *)
    let buft = field t "buft" (ptr void) (* Placeholder: GGML.backend_buffer_type_t *)
    let () = seal t
  end

  (** Model parameters structure. *)
  module ModelParams = struct
    type t

    let t : t structure typ = structure (ns "model_params")

    (* TODO: Define ggml_backend_dev_t in ggml bindings *)
    let devices = field t "devices" (ptr void) (* Placeholder: ptr GGML.backend_dev_t *)
    let tensor_buft_overrides = field t "tensor_buft_overrides" (ptr ModelTensorBuftOverride.t)
    let n_gpu_layers = field t "n_gpu_layers" int32_t
    let split_mode = field t "split_mode" split_mode
    let main_gpu = field t "main_gpu" int32_t
    let tensor_split = field t "tensor_split" (ptr float)
    let progress_callback = field t "progress_callback" progress_callback
    let progress_callback_user_data = field t "progress_callback_user_data" (ptr void)
    let kv_overrides = field t "kv_overrides" (ptr ModelKvOverride.t)
    let vocab_only = field t "vocab_only" bool
    let use_mmap = field t "use_mmap" bool
    let use_mlock = field t "use_mlock" bool
    let check_tensors = field t "check_tensors" bool
    let () = seal t
  end

  type cgraph = [ `Cgraph ] structure typ
  (** The computation graph structure. *)

  let cgraph_struct : cgraph = structure (ns "cgraph")

  (** Pointer to the computation graph structure. *)
  let cgraph = ptr cgraph_struct

  (** Callback function for graph computation. *)
  let graph_compute_callback_type = ptr void @-> cgraph @-> bool @-> returning bool

  let graph_compute_callback = static_funptr graph_compute_callback_type

  (** Callback function for evaluating the computation graph. *)
  let cgraph_eval_callback = static_funptr (cgraph @-> ptr void @-> returning bool)

  (** Callback function to signal an abort condition. *)
  let abort_callback = static_funptr (ptr void @-> returning bool)

  (** Context parameters structure. This structure contains various parameters that control the behavior of the context,
      such as the number of threads, batch size, and other configuration options. *)
  module ContextParams = struct
    type t

    (** The type of the context parameters structure. *)
    let t : t structure typ = structure (ns "context_params")

    (** Number of context tokens. *)
    let n_ctx = field t "n_ctx" uint32_t

    (** Number of tokens in a batch. *)
    let n_batch = field t "n_batch" uint32_t

    (** Number of micro-batches. *)
    let n_ubatch = field t "n_ubatch" uint32_t

    (** Maximum number of sequences. *)
    let n_seq_max = field t "n_seq_max" uint32_t

    (** Number of threads to use. *)
    let n_threads = field t "n_threads" int32_t

    (** Number of threads to use for batch processing. *)
    let n_threads_batch = field t "n_threads_batch" int32_t

    (** Type of rope scaling to use. *)
    let rope_scaling_type = field t "rope_scaling_type" rope_scaling_type

    (** Type of pooling to use. *)
    let pooling_type = field t "pooling_type" pooling_type

    (** Type of attention mechanism to use. *)
    let attention_type = field t "attention_type" attention_type

    (** Base frequency for rope embeddings. *)
    let rope_freq_base = field t "rope_freq_base" float

    (** Scaling factor for rope embeddings. *)
    let rope_freq_scale = field t "rope_freq_scale" float

    (** External factor for YARN scaling. *)
    let yarn_ext_factor = field t "yarn_ext_factor" float

    (** Attention factor for YARN scaling. *)
    let yarn_attn_factor = field t "yarn_attn_factor" float

    (** Fast beta parameter for YARN scaling. *)
    let yarn_beta_fast = field t "yarn_beta_fast" float

    (** Slow beta parameter for YARN scaling. *)
    let yarn_beta_slow = field t "yarn_beta_slow" float

    (** Original context size for YARN scaling. *)
    let yarn_orig_ctx = field t "yarn_orig_ctx" uint32_t

    (** Threshold for defragmentation. *)
    let defrag_thold = field t "defrag_thold" float

    (** Callback function for evaluating the computation graph. *)
    let cb_eval = field t "cb_eval" cgraph_eval_callback

    (** User data for the evaluation callback. *)
    let cb_eval_user_data = field t "cb_eval_user_data" (ptr void)

    (** Type of key tensors. *)
    let type_k = field t "type_k" ggml_type

    (** Type of value tensors. *)
    let type_v = field t "type_v" ggml_type

    (** Whether to compute logits for all tokens. *)
    let logits_all = field t "logits_all" bool

    (** Whether to compute embeddings. *)
    let embeddings = field t "embeddings" bool

    (** Whether to offload key/query/value tensors. *)
    let offload_kqv = field t "offload_kqv" bool

    (** Whether to use flash attention. *)
    let flash_attn = field t "flash_attn" bool

    (** Whether to disable performance optimizations. *)
    let no_perf = field t "no_perf" bool

    (** Callback function to signal an abort condition. *)
    let abort_callback = field t "abort_callback" abort_callback

    (** User data for the abort callback. *)
    let abort_callback_data = field t "abort_callback_data" (ptr void)

    let graph_callback = field t "graph_callback" graph_compute_callback
    let graph_callback_data = field t "graph_callback_data" (ptr void)
    let () = seal t
  end

  (** Model quantization parameters structure. *)
  module ModelQuantizeParams = struct
    type t

    let t : t structure typ = structure (ns "model_quantize_params")
    let nthread = field t "nthread" int32_t
    let ftype = field t "ftype" ftype
    let output_tensor_type = field t "output_tensor_type" ggml_type
    let token_embedding_type = field t "token_embedding_type" ggml_type
    let allow_requantize = field t "allow_requantize" bool
    let quantize_output_tensor = field t "quantize_output_tensor" bool
    let only_copy = field t "only_copy" bool
    let pure = field t "pure" bool
    let keep_split = field t "keep_split" bool
    let imatrix = field t "imatrix" (ptr void)
    let kv_overrides = field t "kv_overrides" (ptr void) (* Assuming C void* maps to vector *)
    let tensor_types = field t "tensor_types" (ptr void) (* Assuming C void* maps to vector *)
    let () = seal t
  end

  (** Logit bias structure. *)
  module LogitBias = struct
    type t

    let t : t structure typ = structure (ns "logit_bias")
    let token = field t "token" token
    let bias = field t "bias" float
    let () = seal t
  end

  (** Sampler chain parameters structure. *)
  module SamplerChainParams = struct
    type t

    let t : t structure typ = structure (ns "sampler_chain_params")
    let no_perf = field t "no_perf" bool
    let () = seal t
  end

  (** Chat message structure. *)
  module ChatMessage = struct
    type t

    let t : t structure typ = structure (ns "chat_message")
    let role = field t "role" string (* const char * *)
    let content = field t "content" string (* const char * *)
    let () = seal t
  end

  (** KV cache view cell structure. *)
  module KvCacheViewCell = struct
    type t

    let t : t structure typ = structure (ns "kv_cache_view_cell")
    let pos = field t "pos" pos
    let () = seal t
  end

  (** KV cache view structure. *)
  module KvCacheView = struct
    type t

    let t : t structure typ = structure (ns "kv_cache_view")
    let n_cells = field t "n_cells" int32_t
    let n_seq_max = field t "n_seq_max" int32_t
    let token_count = field t "token_count" int32_t
    let used_cells = field t "used_cells" int32_t
    let max_contiguous = field t "max_contiguous" int32_t
    let max_contiguous_idx = field t "max_contiguous_idx" int32_t
    let cells = field t "cells" (ptr KvCacheViewCell.t)
    let cells_sequences = field t "cells_sequences" (ptr seq_id)
    let () = seal t
  end

  let struct_sampler : [ `Sampler ] structure typ = structure (ns "sampler")
  let sampler = ptr struct_sampler

  (** Sampler interface structure (function pointers). *)
  module SamplerI = struct
    type t

    let t : t structure typ = structure (ns "sampler_i")
    let name_fn = field t "name" (static_funptr (sampler @-> returning string))
    let accept_fn = field t "accept" (static_funptr (sampler @-> token @-> returning void))
    let apply_fn = field t "apply" (static_funptr (sampler @-> ptr TokenDataArray.t @-> returning void))
    let reset_fn = field t "reset" (static_funptr (sampler @-> returning void))
    let clone_fn = field t "clone" (static_funptr (sampler @-> returning sampler))
    let free_fn = field t "free" (static_funptr (sampler @-> returning void))
    let () = seal t
  end

  (** Sampler structure. *)
  module Sampler = struct
    let t = struct_sampler
    let context_t = typedef (ptr void) (ns "sampler_context_t")

    (* let t : t structure typ = structure (ns "sampler") - Defined as opaque above *)
    let iface = field t "iface" (ptr SamplerI.t)
    let ctx = field t "ctx" context_t
    let () = seal t
  end

  (** Performance context data structure. *)
  module PerfContextData = struct
    type t

    let t : t structure typ = structure (ns "perf_context_data")
    let t_start_ms = field t "t_start_ms" double
    let t_load_ms = field t "t_load_ms" double
    let t_p_eval_ms = field t "t_p_eval_ms" double
    let t_eval_ms = field t "t_eval_ms" double
    let n_p_eval = field t "n_p_eval" int32_t
    let n_eval = field t "n_eval" int32_t
    let () = seal t
  end

  (** Performance sampler data structure. *)
  module PerfSamplerData = struct
    type t

    let t : t structure typ = structure (ns "perf_sampler_data")
    let t_sample_ms = field t "t_sample_ms" double
    let n_sample = field t "n_sample" int32_t
    let () = seal t
  end
end
