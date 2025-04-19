open Ctypes
module Types = Types_generated (* Module containing types generated from yolo_type_description.ml *)

module Functions (F : Ctypes.FOREIGN) = struct
  open F

  (** [yolo_model_init model fname] initializes a YOLO model.
      @param model Pointer to the yolo_model_t structure to initialize.
      @param fname Path to the model file.
      @return 0 on success, non-zero on failure. *)
  let yolo_model_init = foreign "yolo_model_init" (ptr Types.model_t @-> string @-> returning int)

  (** [yolo_model_uninit model] frees resources associated with the YOLO model.
      @param model Pointer to the yolo_model_t structure to uninitialize. *)
  let yolo_model_uninit = foreign "yolo_model_uninit" (ptr Types.model_t @-> returning void)

  (** [yolo_model_graph model n_past n_tokens] builds the computation graph for the YOLO model.
      @param model Pointer to the initialized yolo_model_t structure.
      @param n_past Number of past tokens (likely unused in YOLO, check C++ usage).
      @param n_tokens Number of tokens to process (likely unused in YOLO, check C++ usage).
      @return Pointer to the ggml_cgraph computation graph. *)
  let yolo_model_graph =
    foreign "yolo_model_graph" (ptr Types.model_t @-> int @-> int @-> returning Ggml.Types_generated.cgraph)
end
