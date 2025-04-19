open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "magika_" ^ name

  (* Opaque struct type for the internal buffer *)
  let model_buf : [ `magika_model_buf ] structure typ = structure (ns "model_buf")
  (* Note: We don't seal magika_model_buf as its definition is internal to the C library *)

  (* Struct type for the magika model handle *)
  let model_t : [ `magika_model_t ] structure typ = structure (ns "model_t")
  let buf = field model_t "buf" (ptr model_buf)
  let () = seal model_t
end
