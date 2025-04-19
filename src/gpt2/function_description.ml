open Ctypes
module Types = Types_generated (* Module containing types generated from gpt2_type_description.ml *)

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types

  (** [model_init model fname n_ctx n_gpu_layers] initializes a GPT-2 model.
      @param model Pointer to the model structure to initialize.
      @param fname Path to the model file.
      @param n_ctx Context size.
      @param n_gpu_layers Number of layers to offload to GPU.
      @return 0 on success, non-zero on failure. *)
  let model_init = foreign (ns "model_init") (ptr Types.model_t @-> string @-> int @-> int @-> returning int)

  (** [model_uninit model] frees resources associated with the model.
      @param model Pointer to the model structure to uninitialize. *)
  let model_uninit = foreign (ns "model_uninit") (ptr Types.model_t @-> returning void)

  (** [model_graph model n_past n_tokens] builds the computation graph.
      @param model Pointer to the initialized model structure.
      @param n_past Number of past tokens.
      @param n_tokens Number of tokens to process.
      @return Pointer to the computation graph. *)
  let model_graph =
    foreign (ns "model_graph") (ptr Types.model_t @-> int @-> int @-> returning Ggml.Types_generated.cgraph)
end
