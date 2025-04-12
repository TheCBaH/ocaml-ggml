let%expect_test "constants" =
  Printf.printf "max_dims: %d\n" Ggml_const.C.Types.max_dims;
  [%expect "max_dims: 4"];
  ()
