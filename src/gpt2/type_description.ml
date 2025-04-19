open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "gpt2_" ^ name

  (* Opaque struct type for the internal buffer *)
  let model_buf : [ `gpt2_model_buf ] structure typ = structure (ns "model_buf")
  (* Note: We don't seal gpt2_model_buf as its definition is internal to the C library *)

  (* Struct type for the GPT-2 model handle *)
  let model_t : [ `gpt2_model_t ] structure typ = structure (ns "model_t")
  let buf = field model_t "buf" (ptr model_buf)
  let () = seal model_t
end
