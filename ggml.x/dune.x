(library
 (name ggml)
 (public_name ggml)
 (synopsis "OCaml bindings for ggml")
 (libraries ctypes ctypes.foreign) ; Ensure ctypes.foreign is listed
 (modules ggml_bindings ggml_generated ggml_types) ; Add ggml_types back, it will be generated by the rule below

 ; Define the foreign stubs using the generated C file
 (foreign_stubs
  (language c)
  (names ggml_stubs) ; Base name matches the generated C file target
  ; Flags for compiling the generated C stubs
  (flags (:standard -I../ggml/include -L../ggml/build/src -lggml))
  ; Dependencies for the C stubs compilation
  (extra_deps ../ggml/include/ggml.h ../ggml/build/src/libggml.a ggml_stubs.c) ; Depend on header, libggml, and generated C file
 ))
; Rule to copy the generated types from the generator subdirectory
(rule
 (target ggml_types.ml)
 (deps generator/ggml_types_generated.ml)
 (action (copy %{deps} %{target})))

; Rule to generate the ML bindings module from the functor
(rule
 (target ggml_generated.ml)
 (deps ggml_bindings.ml ../ggml/include/ggml.h) ; Depend on bindings definition and header
 (action
  (run ctypes-gen --ml %{target} -p ggml --functor Ggml_bindings.Make))) ; Call ctypes-gen directly

; Rule to generate the C stubs file from the functor
(rule
 (target ggml_stubs.c)
 (deps ggml_bindings.ml ../ggml/include/ggml.h ../ggml/build/src/libggml.a) ; Depend on bindings, header, and libggml.a
 (action
  (run ctypes-gen --c %{target} -p ggml --functor Ggml_bindings.Make))) ; Call ctypes-gen directly