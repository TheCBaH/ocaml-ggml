open Ctypes
open Ggml.C

let getfp p field = !@(p |-> field)
let to_string t = coerce (ptr char) string t
let pp_int64 fmt t = Format.fprintf fmt "%Ld" t
let pp_list p fmt t = Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@,") p) t)
let pp_pair p1 p2 fmt (t1, t2) = Format.fprintf fmt "@[%a,@,%a@]" p1 t1 p2 t2

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

  let compare a b = Int.compare a.id b.id

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
      let ptr = raw_address_of_ptr @@ to_voidp tensor in
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
          let ptr = raw_address_of_ptr @@ to_voidp tensor in
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
    else Format.(pp_print_list ~pp_sep:pp_print_newline pp) fmt @@ List.sort compare @@ List.map snd nodes

  let of_graph graph =
    let count = Functions.graph_n_nodes graph in
    let rec of_graph_aux nodes n =
      if n < count then
        let t = Functions.graph_node graph n in
        let nodes = add_node n t nodes in
        of_graph_aux nodes @@ succ n
      else nodes
    in
    of_graph_aux (empty count) 0

  let get_id nodes tensor =
    let ptr = raw_address_of_ptr @@ to_voidp tensor in
    PtrMap.find ptr nodes.map

  let fold f nodes a =
    PtrMap.fold
      (fun ptr id a ->
        let tensor = from_voidp Types.Tensor.t @@ ptr_of_raw_address ptr in
        f tensor id a)
      nodes.map a
end

let attr key value = Model_explorer.KeyValue.create ~key ~value

let tensor nodes t =
  let id = TensorId.get_id nodes t in
  let tensor_index = attr "tensor_index" @@ string_of_int id.id in
  let tensor_shape =
    let type_name = Ggml.C.Functions.type_name @@ getfp t Ggml.C.Types.Tensor.typ_ in
    let shape = Format.asprintf "@[%s:%a@]" type_name pp_shape t in
    attr "tensor_shape" shape
  in
  let tensor = [ tensor_index; tensor_shape ] in
  let tensor =
    let name = getfp t Ggml.C.Types.Tensor.name in
    let name = to_string @@ CArray.start name in
    if String.length name == 0 || String.starts_with ~prefix:"leaf_" name || String.starts_with ~prefix:"node_" name
    then tensor
    else attr "tensor_name" name :: tensor
  in
  tensor
