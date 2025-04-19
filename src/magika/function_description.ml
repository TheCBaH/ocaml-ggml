open Ctypes
module Types = Types_generated (* Module containing types generated from magika_type_description.ml *)

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  let ns s = "magika_" ^ s

  (** [magika_model_init model fname] initializes a magika model.
      @param model Pointer to the magika_model_t structure to initialize.
      @param fname Path to the model file.
      @return 0 on success, non-zero on failure. *)
  let model_init = foreign (ns "model_init") (ptr Types.model_t @-> string @-> returning int)

  (** [magika_model_uninit model] frees resources associated with the magika model.
      @param model Pointer to the magika_model_t structure to uninitialize. *)
  let model_uninit = foreign (ns "model_uninit") (ptr Types.model_t @-> returning void)

  (** [magika_model_graph model n_past n_tokens] builds the computation graph for the magika model.
      @param model Pointer to the initialized magika_model_t structure.
      @return Pointer to the ggml_cgraph computation graph. *)
  let model_graph = foreign (ns "model_graph") (ptr Types.model_t @-> returning Ggml.Types_generated.cgraph)
end
