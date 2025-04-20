open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated

  (** Retrieve the default parameters for a model.
      @return A struct containing the default parameters for a model. *)
  let model_default_params = foreign (ns "model_default_params") (void @-> returning ModelParams.t)

  (** Retrieve the default parameters for a context.
      @return A struct containing the default parameters for a context. *)
  let context_default_params = foreign (ns "context_default_params") (void @-> returning ContextParams.t)

  (** Retrieve the default parameters for a sampler chain.
      @return A struct containing the default parameters for a sampler chain. *)
  let sampler_chain_default_params =
    foreign (ns "sampler_chain_default_params") (void @-> returning SamplerChainParams.t)

  (** Retrieve the default parameters for model quantization.
      @return A struct containing the default parameters for model quantization. *)
  let model_quantize_default_params =
    foreign (ns "model_quantize_default_params") (void @-> returning ModelQuantizeParams.t)

  (** Initialize the backend. Call this function once at the start of the program to initialize the llama + ggml
      backend. *)
  let backend_init = foreign (ns "backend_init") (void @-> returning void)

  (** Free the backend resources. Call this function once at the end of the program to free backend resources. Currently
      used for MPI. *)
  let backend_free = foreign (ns "backend_free") (void @-> returning void)

  (** Initialize NUMA (Non-Uniform Memory Access) with a given strategy.
      @param numa_strategy The NUMA strategy to use. *)
  let numa_init = foreign (ns "numa_init") (Ggml.C.Types.CPU.numa_strategy @-> returning void)

  (** Attach a threadpool to a context.
      @param ctx The llama context to attach the threadpool to.
      @param threadpool The threadpool to attach.
      @param threadpool_batch The threadpool for batch processing. *)
  let attach_threadpool =
    foreign (ns "attach_threadpool")
      (context @-> ptr Ggml.C.Types.threadpool @-> ptr Ggml.C.Types.threadpool @-> returning void)

  (** Detach a threadpool from a context.
      @param ctx The llama context to detach the threadpool from. *)
  let detach_threadpool = foreign (ns "detach_threadpool") (context @-> returning void)

  (* Model loading *)

  (** Load the model from a file. If the file is split into multiple parts, the file name must follow this pattern:
      `<name>-%05d-of-%05d.gguf`. If the split file name does not follow this pattern, use {!model_load_from_splits}.
      @param path_model The path to the model file.
      @param params The parameters for loading the model.
      @return A pointer to the loaded model, or NULL on failure. *)
  let model_load_from_file = foreign (ns "model_load_from_file") (string @-> ModelParams.t @-> returning model)

  (** Load the model from multiple splits (supports custom naming scheme). The paths must be in the correct order.
      @param paths A pointer to an array of file paths for the model splits.
      @param n_paths The number of paths in the array.
      @param params The parameters to use when loading the model.
      @return A pointer to the loaded model, or NULL on failure. *)
  let model_load_from_splits =
    foreign (ns "model_load_from_splits") (ptr string @-> size_t @-> ModelParams.t @-> returning model)

  (** Free the memory associated with a model.
      @param model The model to free. *)
  let model_free = foreign (ns "model_free") (model @-> returning void)

  (* Context creation/destruction *)

  (** Initialize a context from a model.
      @param model The model to initialize the context from.
      @param params The parameters to use for the context.
      @return A pointer to the initialized context. *)
  let init_from_model = foreign (ns "init_from_model") (model @-> ContextParams.t @-> returning context)

  (** Free the memory associated with a context.
      @param ctx The context to free. *)
  let free = foreign (ns "free") (context @-> returning void)

  (* Time and system info *)

  (** Get the current time in microseconds.
      @return The current time in microseconds. *)
  let time_us = foreign (ns "time_us") (void @-> returning int64_t)

  (** Get the maximum number of devices supported.
      @return The maximum number of devices supported. *)
  let max_devices = foreign (ns "max_devices") (void @-> returning size_t)

  (** Check if memory-mapped files are supported.
      @return True if memory-mapped files are supported, false otherwise. *)
  let supports_mmap = foreign (ns "supports_mmap") (void @-> returning bool)

  (** Check if memory locking is supported.
      @return True if memory locking is supported, false otherwise. *)
  let supports_mlock = foreign (ns "supports_mlock") (void @-> returning bool)

  (** Check if GPU offloading is supported.
      @return True if GPU offloading is supported, false otherwise. *)
  let supports_gpu_offload = foreign (ns "supports_gpu_offload") (void @-> returning bool)

  (** Check if RPC is supported.
      @return True if RPC is supported, false otherwise. *)
  let supports_rpc = foreign (ns "supports_rpc") (void @-> returning bool)

  (* Context properties *)

  (** Get the context size (number of tokens).
      @param ctx The context.
      @return The context size. *)
  let n_ctx = foreign (ns "n_ctx") (context @-> returning uint32_t)

  (** Get the logical maximum batch size.
      @param ctx The context.
      @return The logical maximum batch size. *)
  let n_batch = foreign (ns "n_batch") (context @-> returning uint32_t)

  (** Get the physical maximum batch size.
      @param ctx The context.
      @return The physical maximum batch size. *)
  let n_ubatch = foreign (ns "n_ubatch") (context @-> returning uint32_t)

  (** Get the maximum number of sequences.
      @param ctx The context.
      @return The maximum number of sequences. *)
  let n_seq_max = foreign (ns "n_seq_max") (context @-> returning uint32_t)

  (* Model properties *)

  (** Get the model associated with a context. *)
  let get_model = foreign (ns "get_model") (context @-> returning const_model)

  (** Get the KV cache associated with a context. *)
  let get_kv_self = foreign (ns "get_kv_self") (context @-> returning kv_cache)

  (** Get the pooling type of the context. TODO: rename to llama_get_pooling_type *)
  let pooling_type = foreign (ns "pooling_type") (context @-> returning pooling_type)

  (** Get the vocabulary associated with a model. *)
  let model_get_vocab = foreign (ns "model_get_vocab") (model @-> returning const_vocab)

  (** Get the RoPE type of the model. *)
  let model_rope_type = foreign (ns "model_rope_type") (model @-> returning rope_type)

  (** Get the training context size of the model. *)
  let model_n_ctx_train = foreign (ns "model_n_ctx_train") (model @-> returning int32_t)

  (** Get the embedding dimension of the model. *)
  let model_n_embd = foreign (ns "model_n_embd") (model @-> returning int32_t)

  (** Get the number of layers in the model. *)
  let model_n_layer = foreign (ns "model_n_layer") (model @-> returning int32_t)

  (** Get the number of attention heads in the model. *)
  let model_n_head = foreign (ns "model_n_head") (model @-> returning int32_t)

  (** Get the number of key/value heads in the model. *)
  let model_n_head_kv = foreign (ns "model_n_head_kv") (model @-> returning int32_t)

  (** Get the model's RoPE frequency scaling factor used during training. *)
  let model_rope_freq_scale_train = foreign (ns "model_rope_freq_scale_train") (model @-> returning float)

  (* Vocab properties *)

  (** Get the type of the vocabulary. *)
  let vocab_type = foreign (ns "vocab_type") (vocab @-> returning vocab_type)

  (** Get the number of tokens in the vocabulary. *)
  let vocab_n_tokens = foreign (ns "vocab_n_tokens") (vocab @-> returning int32_t)

  (* Model metadata *)

  (** Get metadata value as a string by key name. Functions to access the model's GGUF metadata scalar values.
      - The functions return the length of the string on success, or -1 on failure.
      - The output string is always null-terminated and cleared on failure.
      - When retrieving a string, an extra byte must be allocated to account for the null terminator.
      - GGUF array values are not supported by these functions.

      @param model The model.
      @param key The metadata key.
      @param buf The buffer to write the value to.
      @param buf_size The size of the buffer.
      @return The length of the string on success, or -1 on failure. *)
  let model_meta_val_str =
    foreign (ns "model_meta_val_str") (model @-> string @-> ptr char @-> size_t @-> returning int32_t)

  (** Get the number of metadata key/value pairs.
      @param model The model.
      @return The number of metadata key/value pairs. *)
  let model_meta_count = foreign (ns "model_meta_count") (model @-> returning int32_t)

  (** Get metadata key name by index.
      @param model The model.
      @param i The index of the metadata key.
      @param buf The buffer to write the key to.
      @param buf_size The size of the buffer.
      @return The length of the string on success, or -1 on failure. *)
  let model_meta_key_by_index =
    foreign (ns "model_meta_key_by_index") (model @-> int32_t @-> ptr char @-> size_t @-> returning int32_t)

  (** Get metadata value as a string by index.
      @param model The model.
      @param i The index of the metadata value.
      @param buf The buffer to write the value to.
      @param buf_size The size of the buffer.
      @return The length of the string on success, or -1 on failure. *)
  let model_meta_val_str_by_index =
    foreign (ns "model_meta_val_str_by_index") (model @-> int32_t @-> ptr char @-> size_t @-> returning int32_t)

  (** Get a string describing the model type.
      @param model The model.
      @param buf The buffer to write the description to.
      @param buf_size The size of the buffer.
      @return The length of the string on success, or -1 on failure. *)
  let model_desc = foreign (ns "model_desc") (model @-> ptr char @-> size_t @-> returning int32_t)

  (** Returns the total size of all the tensors in the model in bytes.
      @param model The model.
      @return The total size of the tensors in bytes. *)
  let model_size = foreign (ns "model_size") (model @-> returning uint64_t)

  (** Get the default chat template. If name is NULL, returns the default chat template.
      @param model The model.
      @param name The name of the chat template (optional).
      @return The chat template string, or NULL if not available. *)
  let model_chat_template = foreign (ns "model_chat_template") (model @-> string @-> returning string)

  (** Returns the total number of parameters in the model.
      @param model The model.
      @return The total number of parameters. *)
  let model_n_params = foreign (ns "model_n_params") (model @-> returning uint64_t)

  (** Returns true if the model contains an encoder that requires [llama_encode()] call.
      @param model The model.
      @return True if the model has an encoder, false otherwise. *)
  let model_has_encoder = foreign (ns "model_has_encoder") (model @-> returning bool)

  (** Returns true if the model contains a decoder that requires [llama_decode()] call.
      @param model The model.
      @return True if the model has a decoder, false otherwise. *)
  let model_has_decoder = foreign (ns "model_has_decoder") (model @-> returning bool)

  (** For encoder-decoder models, this function returns id of the token that must be provided to the decoder to start
      generating output sequence. For other models, it returns -1.
      @param model The model.
      @return The decoder start token ID, or -1. *)
  let model_decoder_start_token = foreign (ns "model_decoder_start_token") (model @-> returning token)

  (** Returns true if the model is recurrent (like Mamba, RWKV, etc.).
      @param model The model.
      @return True if the model is recurrent, false otherwise. *)
  let model_is_recurrent = foreign (ns "model_is_recurrent") (model @-> returning bool)

  (* Model quantization *)

  (** Quantize a model. Returns 0 on success. *)
  let model_quantize =
    foreign (ns "model_quantize") (string @-> string @-> ptr ModelQuantizeParams.t @-> returning uint32_t)

  (* Adapters *)

  (** Load a LoRA adapter from file. *)
  let adapter_lora_init = foreign (ns "adapter_lora_init") (model @-> string @-> returning adapter_lora)

  (** Manually free a LoRA adapter. Note: loaded adapters will be free when the associated model is deleted. *)
  let adapter_lora_free = foreign (ns "adapter_lora_free") (adapter_lora @-> returning void)

  (** Add a loaded LoRA adapter to given context. This will not modify model's weight. *)
  let set_adapter_lora = foreign (ns "set_adapter_lora") (context @-> adapter_lora @-> float @-> returning int32_t)

  (** Remove a specific LoRA adapter from given context. Return -1 if the adapter is not present in the context. *)
  let rm_adapter_lora = foreign (ns "rm_adapter_lora") (context @-> adapter_lora @-> returning int32_t)

  (** Remove all LoRA adapters from given context. *)
  let clear_adapter_lora = foreign (ns "clear_adapter_lora") (context @-> returning void)

  (** Apply a loaded control vector to a llama_context, or if data is NULL, clear the currently loaded vector. n_embd
      should be the size of a single layer's control, and data should point to an n_embd x n_layers buffer starting from
      layer 1. il_start and il_end are the layer range the vector should apply to (both inclusive). See
      llama_control_vector_load in common to load a control vector. *)
  let apply_adapter_cvec =
    foreign (ns "apply_adapter_cvec")
      (context @-> ptr float @-> size_t @-> int32_t @-> int32_t @-> int32_t @-> returning int32_t)

  (* KV Cache *)

  (** Create an empty KV cache view. (use only for debugging purposes) *)
  let kv_cache_view_init = foreign (ns "kv_cache_view_init") (context @-> int32_t @-> returning KvCacheView.t)

  (** Free a KV cache view. (use only for debugging purposes) *)
  let kv_cache_view_free = foreign (ns "kv_cache_view_free") (ptr KvCacheView.t @-> returning void)

  (** Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes) *)
  let kv_cache_view_update = foreign (ns "kv_cache_view_update") (context @-> ptr KvCacheView.t @-> returning void)

  (** Returns the number of tokens in the KV cache (slow, use only for debug) *)
  let kv_self_n_tokens = foreign (ns "kv_self_n_tokens") (context @-> returning int32_t)

  (** Returns the number of used KV cells (i.e. have at least one sequence assigned to them) *)
  let kv_self_used_cells = foreign (ns "kv_self_used_cells") (context @-> returning int32_t)

  (** Clear the KV cache - both cell info is erased and KV data is zeroed *)
  let kv_self_clear = foreign (ns "kv_self_clear") (context @-> returning void)

  (** Removes all tokens that belong to the specified sequence and have positions in [p0, p1).
      Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
      seq_id < 0 : match any sequence. p0 < 0 : [0, p1]. p1 < 0 : [p0, inf). *)
  let kv_self_seq_rm = foreign (ns "kv_self_seq_rm") (context @-> seq_id @-> pos @-> pos @-> returning bool)

  (** Copy all tokens that belong to the specified sequence to another sequence.
      Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence.
      p0 < 0 : [0, p1]. p1 < 0 : [p0, inf). *)
  let kv_self_seq_cp = foreign (ns "kv_self_seq_cp") (context @-> seq_id @-> seq_id @-> pos @-> pos @-> returning void)

  (** Removes all tokens that do not belong to the specified sequence *)
  let kv_self_seq_keep = foreign (ns "kv_self_seq_keep") (context @-> seq_id @-> returning void)

  (** Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1).
      If the KV cache is RoPEd, the KV data is updated accordingly:
        - lazily on next llama_decode()
        - explicitly with llama_kv_self_update().
      p0 < 0 : [0, p1]. p1 < 0 : [p0, inf). *)

  (** Integer division of the positions by factor of `d > 1`.
      If the KV cache is RoPEd, the KV data is updated accordingly:
        - lazily on next llama_decode()
        - explicitly with llama_kv_self_update().
      p0 < 0 : [0, p1]. p1 < 0 : [p0, inf). *)
  let kv_self_seq_add = foreign (ns "kv_self_seq_add") (context @-> seq_id @-> pos @-> pos @-> pos @-> returning void)

  (** Returns the largest position present in the KV cache for the specified sequence *)
  let kv_self_seq_div = foreign (ns "kv_self_seq_div") (context @-> seq_id @-> pos @-> pos @-> int @-> returning void)

  let kv_self_seq_pos_max = foreign (ns "kv_self_seq_pos_max") (context @-> seq_id @-> returning pos)

  (** Defragment the KV cache. This will be applied:
      - lazily on next llama_decode()
      - explicitly with llama_kv_self_update() *)
  let kv_self_defrag = foreign (ns "kv_self_defrag") (context @-> returning void)

  (** Check if the context supports KV cache shifting *)
  let kv_self_can_shift = foreign (ns "kv_self_can_shift") (context @-> returning bool)

  (** Apply the KV cache updates (such as K-shifts, defragmentation, etc.) *)
  let kv_self_update = foreign (ns "kv_self_update") (context @-> returning void)

  (* State / Sessions *)

  (** Returns the *actual* size in bytes of the state (logits, embedding and kv_cache). Only use when saving the state,
      not when restoring it, otherwise the size may be too small. *)
  let state_get_size = foreign (ns "state_get_size") (context @-> returning size_t)

  (** Copies the state to the specified destination address. Destination needs to have allocated enough memory. Returns
      the number of bytes copied. *)
  let state_get_data = foreign (ns "state_get_data") (context @-> ptr uint8_t @-> size_t @-> returning size_t)

  (** Set the state reading from the specified address. Returns the number of bytes read. *)
  let state_set_data = foreign (ns "state_set_data") (context @-> ptr uint8_t @-> size_t @-> returning size_t)

  (** Load session file. *)
  let state_load_file =
    foreign (ns "state_load_file") (context @-> string @-> ptr token @-> size_t @-> ptr size_t @-> returning bool)

  (** Save session file. *)
  let state_save_file = foreign (ns "state_save_file") (context @-> string @-> ptr token @-> size_t @-> returning bool)

  (** Get the exact size needed to copy the KV cache of a single sequence *)
  let state_seq_get_size = foreign (ns "state_seq_get_size") (context @-> seq_id @-> returning size_t)

  (** Copy the KV cache of a single sequence into the specified buffer *)
  let state_seq_get_data =
    foreign (ns "state_seq_get_data") (context @-> ptr uint8_t @-> size_t @-> seq_id @-> returning size_t)

  (** Copy the sequence data (originally copied with {!state_seq_get_data}) into the specified sequence. Returns:
      - Positive: Ok
      - Zero: Failed to load *)
  let state_seq_set_data =
    foreign (ns "state_seq_set_data") (context @-> ptr uint8_t @-> size_t @-> seq_id @-> returning size_t)

  (** Save the state of a specific sequence to a file. *)
  let state_seq_save_file =
    foreign (ns "state_seq_save_file") (context @-> string @-> seq_id @-> ptr token @-> size_t @-> returning size_t)

  (** Load the state of a specific sequence from a file. *)
  let state_seq_load_file =
    foreign (ns "state_seq_load_file")
      (context @-> string @-> seq_id @-> ptr token @-> size_t @-> ptr size_t @-> returning size_t)

  (* Decoding *)

  (** Return batch for single sequence of tokens. The sequence ID will be fixed to 0. The position of the tokens will be
      tracked automatically by {!decode}.

      NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it.
      @param tokens Pointer to the token array.
      @param n_tokens Number of tokens.
      @return A {!Batch.t} struct. *)
  let batch_get_one = foreign (ns "batch_get_one") (ptr token @-> int32_t @-> returning Batch.t)

  (** Allocates a batch of tokens on the heap that can hold a maximum of [n_tokens]. Each token can be assigned up to
      [n_seq_max] sequence ids. The batch has to be freed with {!batch_free}. If [embd] != 0, [llama_batch.embd] will be
      allocated with size of [n_tokens * embd * sizeof(float)]. Otherwise, [llama_batch.token] will be allocated to
      store [n_tokens] {!Types_generated.token}. The rest of the {!Batch.t} members are allocated with size [n_tokens].
      All members are left uninitialized.
      @param n_tokens Maximum number of tokens the batch can hold.
      @param embd
        If 0, allocate [llama_batch.token]. Otherwise, allocate [llama_batch.embd] with size [n_tokens * embd].
      @param n_seq_max Maximum number of sequence IDs per token.
      @return The allocated {!Batch.t}. *)
  let batch_init = foreign (ns "batch_init") (int32_t @-> int32_t @-> int32_t @-> returning Batch.t)

  (** Frees a batch of tokens allocated with {!batch_init}.
      @param batch The batch to free. *)
  let batch_free = foreign (ns "batch_free") (Batch.t @-> returning void)

  (** Processes a batch of tokens with the encoder part of the encoder-decoder model. Stores the encoder output
      internally for later use by the decoder cross-attention layers.
      @param ctx The context.
      @param batch The batch to process.
      @return 0 on success.
      @return < 0 on error. The KV cache state is restored to the state before this call. *)
  let encode = foreign (ns "encode") (context @-> Batch.t @-> returning int32_t)

  (** Process the tokens in the batch. Positive return values do not mean a fatal error, but rather a warning.
      @param ctx The context.
      @param batch The batch to process.
      @return 0 on success.
      @return 1 if could not find a KV slot for the batch (try reducing the size of the batch or increase the context).
      @return < 0 on error. The KV cache state is restored to the state before this call. *)
  let decode = foreign (ns "decode") (context @-> Batch.t @-> returning int32_t)

  (** Set the number of threads used for decoding.
      @param ctx The context.
      @param n_threads The number of threads used for generation (single token).
      @param n_threads_batch The number of threads used for prompt and batch processing (multiple tokens). *)
  let set_n_threads = foreign (ns "set_n_threads") (context @-> int32_t @-> int32_t @-> returning void)

  (** Get the number of threads used for generation of a single token. *)
  let n_threads = foreign (ns "n_threads") (context @-> returning int32_t)

  (** Get the number of threads used for prompt and batch processing (multiple token). *)
  let n_threads_batch = foreign (ns "n_threads_batch") (context @-> returning int32_t)

  (** Set whether the model is in embeddings mode or not. If true, embeddings will be returned but logits will not. *)
  let set_embeddings = foreign (ns "set_embeddings") (context @-> bool @-> returning void)

  (** Set whether to use causal attention or not. If set to true, the model will only attend to the past tokens. *)
  let set_causal_attn = foreign (ns "set_causal_attn") (context @-> bool @-> returning void)

  (** Set whether the model is in warmup mode or not. If true, all model tensors are activated during llama_decode() to
      load and cache their weights. *)
  let set_warmup = foreign (ns "set_warmup") (context @-> bool @-> returning void)

  (** Set abort callback. If it returns true, execution of llama_decode() will be aborted. Currently works only with CPU
      execution. *)
  let set_abort_callback = foreign (ns "set_abort_callback") (context @-> abort_callback @-> ptr void @-> returning void)

  (** Wait until all computations are finished. This is automatically done when using one of the functions below to
      obtain the computation results and is not necessary to call it explicitly in most cases. *)
  let synchronize = foreign (ns "synchronize") (context @-> returning void)

  (** Token logits obtained from the last call to {!decode}. The logits for which llama_batch.logits[i] != 0 are stored
      contiguously in the order they have appeared in the batch. Rows: number of tokens for which llama_batch.logits[i]
      != 0 Cols: n_vocab *)
  let get_logits = foreign (ns "get_logits") (context @-> returning (ptr float))

  (** Logits for the ith token. For positive indices, Equivalent to: llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
      Negative indicies can be used to access logits in reverse order, -1 is the last logit. returns NULL for invalid
      ids. *)
  let get_logits_ith = foreign (ns "get_logits_ith") (context @-> int32_t @-> returning (ptr float))

  (** Get all output token embeddings. when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
      the embeddings for which llama_batch.logits[i] != 0 are stored contiguously in the order they have appeared in the
      batch. shape: [n_outputs*n_embd] Otherwise, returns NULL. *)
  let get_embeddings = foreign (ns "get_embeddings") (context @-> returning (ptr float))

  (** Get the embeddings for the ith token. For positive indices, Equivalent to: llama_get_embeddings(ctx) +
      ctx->output_ids[i]*n_embd Negative indicies can be used to access embeddings in reverse order, -1 is the last
      embedding. shape: [n_embd] (1-dimensional) returns NULL for invalid ids. *)
  let get_embeddings_ith = foreign (ns "get_embeddings_ith") (context @-> int32_t @-> returning (ptr float))

  (** Get the embeddings for a sequence id. Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE when pooling_type ==
      LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence otherwise: float[n_embd] (1-dimensional)
  *)
  let get_embeddings_seq = foreign (ns "get_embeddings_seq") (context @-> seq_id @-> returning (ptr float))

  (* Vocab *)

  (** Get the text representation of a token. *)
  let vocab_get_text = foreign (ns "vocab_get_text") (vocab @-> token @-> returning string)

  (** Get the score of a token. *)
  let vocab_get_score = foreign (ns "vocab_get_score") (vocab @-> token @-> returning float)

  (** Get the attributes of a token. *)
  let vocab_get_attr = foreign (ns "vocab_get_attr") (vocab @-> token @-> returning token_attr)

  (** Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.) *)
  let vocab_is_eog = foreign (ns "vocab_is_eog") (vocab @-> token @-> returning bool)

  (** Identify if Token Id is a control token or a render-able token *)
  let vocab_is_control = foreign (ns "vocab_is_control") (vocab @-> token @-> returning bool)

  (** Get the beginning-of-sentence token ID. *)
  let vocab_bos = foreign (ns "vocab_bos") (vocab @-> returning token)

  (** Get the end-of-sentence token ID. *)
  let vocab_eos = foreign (ns "vocab_eos") (vocab @-> returning token)

  (** Get the end-of-turn token ID. *)
  let vocab_eot = foreign (ns "vocab_eot") (vocab @-> returning token)

  (** Get the sentence separator token ID. *)
  let vocab_sep = foreign (ns "vocab_sep") (vocab @-> returning token)

  (** Get the next-line token ID. *)
  let vocab_nl = foreign (ns "vocab_nl") (vocab @-> returning token)

  (** Get the padding token ID. *)
  let vocab_pad = foreign (ns "vocab_pad") (vocab @-> returning token)

  (** Check if the vocabulary adds a BOS token by default. *)
  let vocab_get_add_bos = foreign (ns "vocab_get_add_bos") (vocab @-> returning bool)

  (** Check if the vocabulary adds an EOS token by default. *)
  let vocab_get_add_eos = foreign (ns "vocab_get_add_eos") (vocab @-> returning bool)

  (** Get the fill-in-the-middle prefix token ID. *)
  let vocab_fim_pre = foreign (ns "vocab_fim_pre") (vocab @-> returning token)

  (** Get the fill-in-the-middle suffix token ID. *)
  let vocab_fim_suf = foreign (ns "vocab_fim_suf") (vocab @-> returning token)

  (** Get the fill-in-the-middle middle token ID. *)
  let vocab_fim_mid = foreign (ns "vocab_fim_mid") (vocab @-> returning token)

  (** Get the fill-in-the-middle padding token ID. *)
  let vocab_fim_pad = foreign (ns "vocab_fim_pad") (vocab @-> returning token)

  (** Get the fill-in-the-middle repeat token ID. *)
  let vocab_fim_rep = foreign (ns "vocab_fim_rep") (vocab @-> returning token)

  (** Get the fill-in-the-middle separator token ID. *)
  let vocab_fim_sep = foreign (ns "vocab_fim_sep") (vocab @-> returning token)

  (* Tokenization *)

  (** Convert the provided text into tokens. The API is thread-safe.
      @param vocab The vocabulary to use.
      @param text The text to tokenize.
      @param text_len The length of the text.
      @param tokens The tokens pointer must be large enough to hold the resulting tokens.
      @param n_tokens_max The maximum number of tokens to write.
      @param add_special Allow adding BOS and EOS tokens if the model is configured to do so.
      @param parse_special
        Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does
        not insert a leading space.
      @return Returns the number of tokens on success (no more than n_tokens_max).
      @return Returns a negative number on failure (-n) where n is the number of tokens that would have been returned.
  *)
  let tokenize =
    foreign (ns "tokenize")
      (vocab @-> string @-> int32_t @-> ptr token @-> int32_t @-> bool @-> bool @-> returning int32_t)

  (** Convert a token ID into its text representation (piece). Uses the vocabulary in the provided context. Does not
      write null terminator to the buffer. User can skip up to 'lstrip' leading spaces before copying (useful when
      encoding/decoding multiple tokens with 'add_space_prefix').
      @param vocab The vocabulary to use.
      @param token The token ID.
      @param buf The buffer to write the piece to.
      @param length The maximum length of the buffer.
      @param lstrip Number of leading spaces to strip.
      @param special If true, special tokens are rendered in the output.
      @return The number of bytes written to the buffer. *)
  let token_to_piece =
    foreign (ns "token_to_piece") (vocab @-> token @-> ptr char @-> int32_t @-> int32_t @-> bool @-> returning int32_t)

  (** Convert the provided tokens into text (inverse of {!tokenize}).
      @param vocab The vocabulary to use.
      @param tokens The array of token IDs.
      @param n_tokens The number of tokens in the array.
      @param text The char pointer must be large enough to hold the resulting text.
      @param text_len_max The maximum length of the output text buffer.
      @param remove_special Allow removing BOS and EOS tokens if the model is configured to do so.
      @param unparse_special If true, special tokens are rendered in the output.
      @return Returns the number of chars/bytes on success (no more than text_len_max).
      @return
        Returns a negative number on failure (-n) where n is the number of chars/bytes that would have been returned. *)
  let detokenize =
    foreign (ns "detokenize")
      (vocab @-> ptr token @-> int32_t @-> ptr char @-> int32_t @-> bool @-> bool @-> returning int32_t)

  (* Chat templates *)

  (** Apply chat template. Inspired by hf apply_chat_template() on python. Both "model" and "custom_template" are
      optional, but at least one is required. "custom_template" has higher precedence than "model". NOTE: This function
      does not use a jinja parser. It only supports a pre-defined list of templates. See more:
      https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
      @param tmpl
        A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used
        instead.
      @param chat Pointer to a list of multiple llama_chat_message.
      @param n_msg Number of llama_chat_message in this chat.
      @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
      @param buf
        A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of
        all messages).
      @param length The size of the allocated buffer.
      @return
        The total number of bytes of the formatted prompt. If it is larger than the size of buffer, you may need to
        re-alloc it and then re-apply the template. *)
  let chat_apply_template =
    foreign (ns "chat_apply_template")
      (string @-> ptr ChatMessage.t @-> size_t @-> bool @-> ptr char @-> int32_t @-> returning int32_t)

  (** Get list of built-in chat templates.
      @param output A pointer to an array of strings to store the template names.
      @param len The maximum number of template names the output array can hold.
      @return The number of built-in chat templates. *)
  let chat_builtin_templates = foreign (ns "chat_builtin_templates") (ptr string @-> size_t @-> returning int32_t)

  (* Sampling API *)

  (** Initialize a sampler from an interface and context. *)
  let sampler_init = foreign (ns "sampler_init") (ptr SamplerI.t @-> ptr void @-> returning sampler)

  (** Get the name of the sampler. *)
  let sampler_name = foreign (ns "sampler_name") (sampler @-> returning string)

  (** Accept a token into the sampler's internal state. *)
  let sampler_accept = foreign (ns "sampler_accept") (sampler @-> token @-> returning void)

  (** Apply the sampler to a token data array. *)
  let sampler_apply = foreign (ns "sampler_apply") (sampler @-> ptr TokenDataArray.t @-> returning void)

  (** Reset the sampler's internal state. *)
  let sampler_reset = foreign (ns "sampler_reset") (sampler @-> returning void)

  (** Clone the sampler. *)
  let sampler_clone = foreign (ns "sampler_clone") (sampler @-> returning sampler)

  (** Free the sampler. Important: do not free if the sampler has been added to a llama_sampler_chain (via
      {!sampler_chain_add}). *)
  let sampler_free = foreign (ns "sampler_free") (sampler @-> returning void)

  (* Sampler Chain *)

  (** Initialize a sampler chain. *)
  let sampler_chain_init = foreign (ns "sampler_chain_init") (SamplerChainParams.t @-> returning sampler)

  (** Add a sampler to the chain. Important: takes ownership of the sampler object and will free it when {!sampler_free}
      is called on the chain. *)
  let sampler_chain_add = foreign (ns "sampler_chain_add") (sampler @-> sampler @-> returning void)

  (** Get a sampler from the chain by index. *)
  let sampler_chain_get = foreign (ns "sampler_chain_get") (sampler @-> int32_t @-> returning sampler)

  (** Get the number of samplers in the chain. *)
  let sampler_chain_n = foreign (ns "sampler_chain_n") (sampler @-> returning int)

  (** Remove a sampler from the chain by index. After removing a sampler, the chain will no longer own it, and it will
      not be freed when the chain is freed. *)
  let sampler_chain_remove = foreign (ns "sampler_chain_remove") (sampler @-> int32_t @-> returning sampler)

  (* Available Samplers *)

  (** Initialize a greedy sampler. *)
  let sampler_init_greedy = foreign (ns "sampler_init_greedy") (void @-> returning sampler)

  (** Initialize a distribution sampler. *)
  let sampler_init_dist = foreign (ns "sampler_init_dist") (uint32_t @-> returning sampler)

  (** Initialize a Top-K sampler. *)
  let sampler_init_top_k = foreign (ns "sampler_init_top_k") (int32_t @-> returning sampler)

  (** Initialize a Top-P (nucleus) sampler. *)
  let sampler_init_top_p = foreign (ns "sampler_init_top_p") (float @-> size_t @-> returning sampler)

  (** Initialize a Min-P sampler. *)
  let sampler_init_min_p = foreign (ns "sampler_init_min_p") (float @-> size_t @-> returning sampler)

  (** Initialize a Locally Typical sampler. *)
  let sampler_init_typical = foreign (ns "sampler_init_typical") (float @-> size_t @-> returning sampler)

  (** Initialize a temperature sampler. Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at
      it's original value, the rest are set to -inf. *)
  let sampler_init_temp = foreign (ns "sampler_init_temp") (float @-> returning sampler)

  (** Initialize a dynamic temperature sampler (a.k.a. entropy). *)
  let sampler_init_temp_ext = foreign (ns "sampler_init_temp_ext") (float @-> float @-> float @-> returning sampler)

  (** Initialize an XTC sampler. *)
  let sampler_init_xtc = foreign (ns "sampler_init_xtc") (float @-> float @-> size_t @-> uint32_t @-> returning sampler)

  (** Initialize a Top-n sigma sampler. *)
  let sampler_init_top_n_sigma = foreign (ns "sampler_init_top_n_sigma") (float @-> returning sampler)

  (** Initialize a Mirostat 1.0 sampler. *)
  let sampler_init_mirostat =
    foreign (ns "sampler_init_mirostat") (int32_t @-> uint32_t @-> float @-> float @-> int32_t @-> returning sampler)

  (** Initialize a Mirostat 2.0 sampler. *)
  let sampler_init_mirostat_v2 =
    foreign (ns "sampler_init_mirostat_v2") (uint32_t @-> float @-> float @-> returning sampler)

  (** Initialize a GBNF grammar sampler. *)
  let sampler_init_grammar = foreign (ns "sampler_init_grammar") (vocab @-> string @-> string @-> returning sampler)

  (** Initialize a lazy GBNF grammar sampler with patterns. *)
  let sampler_init_grammar_lazy_patterns =
    foreign
      (ns "sampler_init_grammar_lazy_patterns")
      (vocab @-> string @-> string @-> ptr string @-> size_t @-> ptr token @-> size_t @-> returning sampler)

  (** Initialize a penalties sampler. NOTE: Avoid using on the full vocabulary as searching for repeated tokens can
      become slow. *)
  let sampler_init_penalties =
    foreign (ns "sampler_init_penalties") (int32_t @-> float @-> float @-> float @-> returning sampler)

  (** Initialize a DRY sampler. *)
  let sampler_init_dry =
    foreign (ns "sampler_init_dry")
      (vocab @-> int32_t @-> float @-> float @-> int32_t @-> int32_t @-> ptr string @-> size_t @-> returning sampler)

  (** Initialize a logit bias sampler. *)
  let sampler_init_logit_bias =
    foreign (ns "sampler_init_logit_bias") (int32_t @-> int32_t @-> ptr LogitBias.t @-> returning sampler)

  (** Initialize an infill sampler. Meant to be used for fill-in-the-middle infilling. *)
  let sampler_init_infill = foreign (ns "sampler_init_infill") (vocab @-> returning sampler)

  (** Returns the seed used by the sampler if applicable, [LLAMA_DEFAULT_SEED] otherwise.
      @param smpl The sampler.
      @return The seed used by the sampler. *)
  let sampler_get_seed = foreign (ns "sampler_get_seed") (sampler @-> returning uint32_t)

  (** Sample and accept a token from the idx-th output of the last evaluation.
      @param smpl The sampler chain.
      @param ctx The llama context.
      @param idx The index of the logit output to sample from (e.g., -1 for the last token).
      @return The sampled token ID. *)
  let sampler_sample = foreign (ns "sampler_sample") (sampler @-> context @-> int32_t @-> returning token)

  (* Model split *)

  (** Build a split GGUF final path for a chunk. Returns the split_path length. *)
  let split_path = foreign (ns "split_path") (ptr char @-> size_t @-> string @-> int @-> int @-> returning int)

  (** Extract the path prefix from the split_path if and only if the split_no and split_count match. Returns the
      split_prefix length. *)
  let split_prefix = foreign (ns "split_prefix") (ptr char @-> size_t @-> string @-> int @-> int @-> returning int)

  (* System info *)

  (** Print system information. *)
  let print_system_info = foreign (ns "print_system_info") (void @-> returning string)

  (** Set callback for all future logging events. If this is not called, or NULL is supplied, everything is output on
      stderr. *)
  let log_set = foreign (ns "log_set") (Ggml.C.Types.log_callback @-> ptr void @-> returning void)

  (* Performance utils *)

  (** Get performance data for the context. NOTE: Used by llama.cpp examples, avoid using in third-party apps. *)
  let perf_context = foreign (ns "perf_context") (context @-> returning PerfContextData.t)

  (** Print performance data for the context. NOTE: Used by llama.cpp examples, avoid using in third-party apps. *)
  let perf_context_print = foreign (ns "perf_context_print") (context @-> returning void)

  (** Reset performance data for the context. NOTE: Used by llama.cpp examples, avoid using in third-party apps. *)
  let perf_context_reset = foreign (ns "perf_context_reset") (context @-> returning void)

  (** Get performance data for the sampler chain. NOTE: Works only with samplers constructed via {!sampler_chain_init}.
  *)
  let perf_sampler = foreign (ns "perf_sampler") (sampler @-> returning PerfSamplerData.t)

  (** Print performance data for the sampler chain. NOTE: Works only with samplers constructed via
      {!sampler_chain_init}. *)
  let perf_sampler_print = foreign (ns "perf_sampler_print") (sampler @-> returning void)

  (** Reset performance data for the sampler chain. NOTE: Works only with samplers constructed via
      {!sampler_chain_init}. *)
  let perf_sampler_reset = foreign (ns "perf_sampler_reset") (sampler @-> returning void)
end
