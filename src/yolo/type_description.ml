open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "yolo_" ^ name

  (* Opaque struct type for the internal buffer *)
  let model_buf : [ `yolo_model_buf ] structure typ = structure (ns "model_buf")
  (* Note: We don't seal yolo_model_buf as its definition is internal to the C library *)

  (* Struct type for the YOLO model handle *)
  let model_t : [ `yolo_model_t ] structure typ = structure (ns "model_t")
  let buf = field model_t "buf" (ptr model_buf)
  let () = seal model_t
end
