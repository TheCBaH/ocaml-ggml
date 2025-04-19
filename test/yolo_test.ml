open Ctypes
open Yolo.C

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let print pp t = Format.printf "@[%a@]@.%!" pp t

let yolo () =
  let model = make Types.model_t in
  let fname = "models/yolov3-tiny.gguf" in
  let rc = Functions.model_init (addr model) fname in
  ignore @@ Functions.model_graph (addr model);
  Functions.model_uninit (addr model);
  assert (rc = 0);
  ()

let%expect_test "yolo" =
  let model = make Types.model_t in
  let fname = "models/yolov3-tiny.gguf" in
  let rc = Functions.model_init (addr model) fname in
  assert (rc = 0);
  [%expect {| |}];

  Functions.model_uninit (addr model);
  keep model;
  ()
