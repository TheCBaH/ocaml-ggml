open Ctypes
open Ggml.C

let getfp p field = !@(p |-> field)
let to_string t = coerce (ptr char) string t
let pp_int64 fmt t = Format.fprintf fmt "%Ld" t
let pp_list p fmt t = Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@,") p) t)
let pp_pair p1 p2 fmt (t1, t2) = Format.fprintf fmt "@[%a,@,%a@]" p1 t1 p2 t2

let pp_shape fmt t =
  let rec cut_aux l' l =
    match l with
    | [] -> l'
    | hd :: tl ->
        let l' = if hd = 1L || hd = 0L then l' else hd :: l' in
        cut_aux l' tl
  in
  let ne = CArray.to_list @@ getfp t Types.Tensor.ne in
  let ne = if true then cut_aux [] @@ List.rev ne else ne in
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

let node_inputs t =
  let src = getfp t Types.Tensor.src in
  List.rev @@ snd
  @@ CArray.fold_left
       (fun (n, l) tensor ->
         let l = if is_null tensor then l else (n, tensor) :: l in
         (succ n, l))
       (0, []) src

module TensorId = struct
  module PtrMap = Map.Make (Nativeint)

  type kind = Input | Output | Constant | Intermediate

  let kind_to_string kind =
    match kind with Input -> "Input" | Output -> "Output" | Constant -> "Constant" | Intermediate -> "Intermediate"

  type t = { id : int; kind : kind }

  let compare a b = Int.compare a.id b.id

  type tensors = { map : t PtrMap.t; node_count : int; next : int }

  let empty node_count = { map = PtrMap.empty; node_count; next = node_count }
  let pp_addr fmt t = Format.fprintf fmt "%#LX" @@ Int64.of_nativeint t
  let pp fmt t = Format.fprintf fmt "@[{id:%d;@ kind:%s}" t.id @@ kind_to_string t.kind

  let add_node id tensor tensors =
    assert (id < tensors.node_count);
    let open Ggml_const.C.Types in
    let t =
      let flags = getfp tensor Types.Tensor.flags in
      let kind = if Int32.logand flags tensor_flag_output = Int32.zero then Intermediate else Output in
      { id; kind }
    in
    let tensors =
      let ptr = raw_address_of_ptr @@ to_voidp tensor in
      { tensors with map = PtrMap.add ptr t tensors.map }
    in
    let inputs = node_inputs tensor in
    List.fold_left
      (fun tensors (_, tensor) ->
        let ptr = raw_address_of_ptr @@ to_voidp tensor in
        if PtrMap.mem ptr tensors.map then
          let _ = if false then Format.eprintf "%d: duplicate ptr:%a@." id pp_addr ptr in
          tensors
        else
          let _ =
            if false then
              Format.eprintf "%d:added :%d %s@." id tensors.next @@ Functions.op_name @@ getfp tensor Types.Tensor.op
          in
          let id = tensors.next in
          let flags = getfp tensor Types.Tensor.flags in
          let kind =
            if Int32.logand flags tensor_flag_param <> Int32.zero then Constant
            else if Int32.logand flags tensor_flag_input <> Int32.zero then Input
            else if getfp tensor Types.Tensor.op = Ggml.Types.Op.None then Constant
            else Intermediate
          in
          let t = { id; kind } in
          let map = PtrMap.add ptr t tensors.map in
          { tensors with map; next = succ tensors.next })
      tensors inputs

  let pp_nodes fmt t =
    let nodes = PtrMap.bindings t.map in
    if false then Format.(pp_print_list ~pp_sep:pp_print_newline (pp_pair pp_addr pp)) fmt nodes
    else Format.(pp_print_list ~pp_sep:pp_print_newline pp) fmt @@ List.sort compare @@ List.map snd nodes

  let of_graph graph =
    let node_count = Functions.graph_n_nodes graph in
    let rec of_graph_aux tensors n =
      if n < node_count then
        let t = Functions.graph_node graph n in
        let tensors = add_node n t tensors in
        of_graph_aux tensors @@ succ n
      else tensors
    in
    of_graph_aux (empty node_count) 0

  let get_id tensors tensor =
    let ptr = raw_address_of_ptr @@ to_voidp tensor in
    PtrMap.find ptr tensors.map

  let tensor_of_ptr ptr = from_voidp Types.Tensor.t @@ ptr_of_raw_address ptr

  let get_tensor tensors id' =
    PtrMap.fold
      (fun ptr id t -> match t with None when id.id = id' -> Some (tensor_of_ptr ptr) | _ as x -> x)
      tensors.map None

  let fold f tensors a =
    PtrMap.fold
      (fun ptr id a ->
        let tensor = tensor_of_ptr ptr in
        f tensor id a)
      tensors.map a
end

let attr key value = Model_explorer.KeyValue.create ~key ~value
let to_id n = string_of_int n

let tensor id t =
  let tensor_index = attr "tensor_index" @@ to_id id.TensorId.id in
  let tensor_shape =
    let type_name = Ggml.C.Functions.type_name @@ getfp t Ggml.C.Types.Tensor.typ_ in
    let shape = Format.asprintf "@[%s%a@]" type_name pp_shape t in
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

let incomingEdge sourceNodeId inputIndex =
  Model_explorer.IncomingEdge.create ~sourceNodeId:(to_id sourceNodeId) ~sourceNodeOutputId:(to_id 0)
    ~targetNodeInputId:(to_id inputIndex) ()

let tensorMetadata index id t =
  let attrs = tensor id t in
  Model_explorer.MetadataItem.create ~id:(to_id index) ~attrs

let node ?id tensors t =
  let id = match id with None -> TensorId.get_id tensors t | Some id -> id in
  let label =
    TensorId.(
      match id.kind with
      | Constant -> "pseudo_const"
      | Input -> "Input"
      | Intermediate | Output ->
          let name = Functions.op_desc t in
          name)
  in
  let inputsMetadata, incomingEdges =
    TensorId.(
      match id.kind with
      | Constant | Input -> (None, None)
      | Intermediate | Output ->
          let inputs = node_inputs t in
          let inputsMetadata, incomingEdges =
            List.fold_left
              (fun (inputs, edges) (index, input) ->
                let id = TensorId.get_id tensors input in
                let edges = incomingEdge id.id index :: edges in
                let inputs = tensorMetadata index id input :: inputs in
                (inputs, edges))
              ([], []) inputs
          in
          (Some (List.rev inputsMetadata), Some (List.rev incomingEdges)))
  in
  let outputsMetadata = [ tensorMetadata 0 id t ] in
  Model_explorer.GraphNode.create ~id:(to_id id.id) ?inputsMetadata ?incomingEdges ~outputsMetadata ~namespace:"" ~label
    ()

let output id =
  let label = "Output" in
  let incomingEdges = [ incomingEdge id.TensorId.id 0 ] in
  Model_explorer.GraphNode.create ~id:(label ^ ":" ^ to_id id.id) ~incomingEdges ~namespace:"" ~label ()

let graph graph =
  let tensors = TensorId.of_graph graph in
  let nodes =
    List.rev
    @@ TensorId.(
         fold (fun t id nodes ->
             let nodes = node tensors ~id t :: nodes in
             if id.kind = Output then output id :: nodes else nodes))
         tensors []
  in
  Model_explorer.Graph.create ~id:"main" ~nodes ()

let visualize ~label cgraph = Model_explorer.GraphCollection.create ~label ~graphs:[ graph cgraph ]
