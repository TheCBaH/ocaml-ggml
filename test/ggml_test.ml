let%expect_test "constants" =
  Printf.printf "max_dims: %d\n" Ggml_const.C.Types.max_dims;
  [%expect "max_dims: 4"];
  ()

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))

open Ctypes
open Ggml.C

let with_params f =
  let params = make Types.InitParams.t in
  setf params Types.InitParams.no_alloc false;
  setf params Types.InitParams.mem_size @@ Unsigned.Size_t.of_int @@ (1024 * 1024);
  setf params Types.InitParams.mem_buffer null;
  let a = f params in
  keep params;
  a

let%expect_test "context" =
  let context = with_params Functions.init in
  let mem = Functions.used_mem context in
  Functions.free context;
  Format.printf "%a%!" Unsigned.Size_t.pp mem;
  [%expect "0"];
  ()

let%expect_test "compute" =
  let context = with_params Functions.init in
  let matrix x y data =
    let len = x * y in
    assert (len = Array.length data);
    Bigarray.Array1.init Bigarray.float32 Bigarray.c_layout len (fun n -> float_of_int @@ Array.get data n)
  in

  let cols_A = 2 in
  let rows_A = 4 in
  let matrix_A = matrix cols_A rows_A [| 2; 8; 5; 1; 4; 2; 8; 6 |] in
  let rows_B = 3 in
  let cols_B = 2 in
  let matrix_B = matrix cols_B rows_B [| 10; 5; 9; 9; 5; 4 |] in
  let set_fp32 tensor matrix =
    let len = Int64.to_int @@ Functions.nelements tensor in
    assert (len = Bigarray.Array1.dim matrix);
    let ptr = !@(tensor |-> Types.Tensor.data) in
    let fp32 = from_voidp float ptr in
    let dst = CArray.from_ptr fp32 len in
    let src = array_of_bigarray array1 matrix in
    CArray.iteri (CArray.set dst) src;
    ()
  in

  let a = Functions.new_tensor_2d context Ggml.Types.Type.F32 (Int64.of_int cols_A) (Int64.of_int rows_A) in
  let b = Functions.new_tensor_2d context Ggml.Types.Type.F32 (Int64.of_int cols_B) (Int64.of_int rows_B) in
  set_fp32 a matrix_A;
  set_fp32 b matrix_B;

  let graph = Functions.new_graph context in
  let result = Functions.mul_mat context a b in
  Functions.build_forward_expand graph result;

  (* Functions.graph_co *)
  Functions.free context;
  [%expect {||}];
  ()
