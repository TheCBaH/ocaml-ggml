let () =
  let generate_ml_file = "ggml_types_generated.ml" in
  let header = Sys.argv.(1) in
  let prefix = "ggml_stub" in

  let oc = open_out generate_ml_file in
  let pp_fmt = Format.formatter_of_out_channel oc in

  (* Configure CFLAGS to include the ggml header directory *)
  let cflags = ["-I"; Filename.dirname header] in

  (* Generate ML code for types, enums, and constants *)
  Cstubs_structs.write_ml
    ~concurrency:Cstubs.unlocked
    ~errno:Cstubs.ignore_errno
    ~cflags
    pp_fmt
    (module Ggml_bindings.Def); (* Use Def from ggml_bindings.ml *)

  (* Generate C stubs (optional here, could be done in main library dune file) *)
  (* We'll let the main library dune file handle C stub generation *)
  (*
  let generate_c_file = "ggml_stubs.c" in
  let () = Out_channel.with_open_text generate_c_file (fun oc ->
    let fmt = Format.formatter_of_out_channel oc in
    Format.fprintf fmt "#include \"%s\"@." (Filename.basename header);
    Cstubs_structs.write_c ~cflags fmt (module Ggml_bindings.Def)
  )
  *)

  Format.pp_print_flush pp_fmt ();
  close_out oc

(* The Ggml_bindings module is now defined in ../ggml_bindings.ml *)
(* We just need to open it if necessary, or qualify the Def module *)

(* No need to redefine Ggml_bindings here *)