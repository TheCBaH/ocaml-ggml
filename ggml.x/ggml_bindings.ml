(* ggml-ocaml/ggml_bindings.ml *)
open Ctypes
open Foreign

(* This module defines types, constants, and enums based on ggml.h *)
(* It will be used by both the generator and the stub generation *)
module Def (F : Ctypes.TYPE) = struct
  (* We will add type/enum/constant definitions here later, *)
  (* potentially by including the generated ggml_types.ml *)
  let () = () (* Placeholder *)
end

(* This functor defines the actual C function bindings *)
(* It's used by ctypes.foreign via the dune file to generate C stubs *)
module Make (F : Cstubs.FOREIGN) = struct
  (* We'll add actual function bindings here later *)
  (* Example: let ggml_init = F.foreign "ggml_init" (void @-> returning void) *)
  let () = () (* Placeholder *)
end