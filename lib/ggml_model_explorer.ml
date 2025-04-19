open Ctypes
open Ggml.C

let getfp p field = !@(p |-> field)
let to_string t = coerce (ptr char) string t
let pp_int64 fmt t = Format.fprintf fmt "%Ld" t
let pp_list p fmt t = Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@ ") p) t)
let pp_pair p1 p2 fmt (t1, t2) = Format.fprintf fmt "@[%a,@ %a@]" p1 t1 p2 t2

(* let attr key value = KeyValue.create ~key ~value *)

let pp_shape fmt t =
  let rec cut_aux l' l =
    match l with [] -> l' | hd :: _ when hd = 1L || hd = 0L -> l' | hd :: tl -> cut_aux (hd :: l') tl
  in
  let ne = List.rev @@ cut_aux [] @@ CArray.to_list @@ getfp t Types.Tensor.ne in
  pp_list pp_int64 fmt ne

let pp_flags fmt t =
  let flags = getfp t Types.Tensor.flags in
  let add name c l = if Int32.logand flags c = Int32.zero then l else name :: l in
  let flags =
    let open Ggml_const.C.Types in
    [] |> add "Input" tensor_flag_input |> add "Output" tensor_flag_output |> add "Param" tensor_flag_param
    |> add "Loss" tensor_flag_loss
  in
  pp_list Format.pp_print_string fmt flags

module TensorId = struct
  module PtrMap = Map.Make (Nativeint)

  type kind = Input | Output | Constant | Intermediate

  let kind_to_string kind =
    match kind with Input -> "Input" | Output -> "Output" | Constant -> "Constant" | Intermediate -> "Intermediate"

  type t = { id : int; kind : kind }
  type nodes = { map : t PtrMap.t; node_count : int; next : int }

  let empty node_count = { map = PtrMap.empty; node_count; next = node_count }
  let pp_addr fmt t = Format.fprintf fmt "%#LX" @@ Int64.of_nativeint t
  let pp fmt t = Format.fprintf fmt "@[{id:%d;@ kind:%s}" t.id @@ kind_to_string t.kind

  let add_node id tensor nodes =
    assert (id < nodes.node_count);
    let open Ggml_const.C.Types in
    let t =
      let flags = getfp tensor Types.Tensor.flags in
      let kind = if Int32.logand flags tensor_flag_output = Int32.zero then Intermediate else Output in
      { id; kind }
    in
    let nodes =
      let ptr = Ctypes.raw_address_of_ptr @@ to_voidp tensor in
      { nodes with map = PtrMap.add ptr t nodes.map }
    in
    let src = getfp tensor Types.Tensor.src in
    let () =
      let l = CArray.to_list src |> List.filter (fun t -> not @@ is_null t) in
      if false then Format.eprintf "%d:not-null:%d@." id @@ List.length l;
      ()
    in
    CArray.fold_left
      (fun nodes tensor ->
        if is_null tensor then nodes
        else
          let ptr = Ctypes.raw_address_of_ptr @@ to_voidp tensor in
          if PtrMap.mem ptr nodes.map then
            let _ = if false then Format.eprintf "%d: duplicate ptr:%a@." id pp_addr ptr in
            nodes
          else
            let _ =
              if false then
                Format.eprintf "%d:added :%d %s@." id nodes.next @@ Functions.op_name @@ getfp tensor Types.Tensor.op
            in
            let id = nodes.next in
            let flags = getfp tensor Types.Tensor.flags in
            let kind =
              if Int32.logand flags tensor_flag_param <> Int32.zero then Constant
              else if Int32.logand flags tensor_flag_input <> Int32.zero then Input
              else if getfp tensor Types.Tensor.op = Ggml.Types.Op.None then Constant
              else Intermediate
            in
            let t = { id; kind } in
            let map = PtrMap.add ptr t nodes.map in
            { nodes with map; next = succ nodes.next })
      nodes src

  let pp_nodes fmt t =
    let nodes = PtrMap.bindings t.map in
    if false then Format.(pp_print_list ~pp_sep:pp_print_newline (pp_pair pp_addr pp)) fmt nodes
    else Format.(pp_print_list ~pp_sep:pp_print_newline pp) fmt @@ List.map snd nodes

  let of_graph graph =
    let nodes = Array.init (Functions.graph_n_nodes graph) (fun n -> Functions.graph_node graph n) in
    let tensors = ref @@ empty @@ Array.length nodes in
    Array.iteri (fun id t -> tensors := add_node id t !tensors) nodes;
    !tensors
end
