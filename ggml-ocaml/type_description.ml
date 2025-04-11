(* open Ctypes *)

module Types (F : Ctypes.TYPE) = struct
  open F

  let max_dims = constant "GGML_MAX_DIMS" int
end