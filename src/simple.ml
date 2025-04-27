open Ctypes
open Cmdliner
open Cmdliner.Term.Syntax

let print_graph ~label graph out_file =
  let graph = Ggml_model_explorer.visualize ~label graph in
  let oc = open_out out_file in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      Result.get_ok
      @@ Jsont_bytesrw.encode ~eod:true ~format:Jsont.Indent Model_explorer.GraphCollection.jsont graph
      @@ Bytesrw.Bytes.Writer.of_out_channel oc);
  ignore (graph, out_file)

module GraphKey = struct
  type tensor = { typ : Ggml.Types.Type.t; shape : int list }
  type tensors = tensor list
  type t = { nodes : int; inputs : tensors; outputs : tensors }

  let make nodes inputs outputs =
    let tensors = List.map (fun (typ, shape) -> { typ; shape }) in
    { nodes; inputs = tensors inputs; outputs = tensors outputs }

  let pp_sep ch = fun fmt () -> Format.fprintf fmt "%c@ " ch
  let coma = pp_sep ','
  let semicolon = pp_sep ';'

  let pp_tensor fmt { typ; shape } =
    Format.(
      fprintf fmt "@[{%s:@,[%a]}@]" (Ggml.Types.Type.to_string typ) (pp_print_list ~pp_sep:coma pp_print_int) shape)

  let pp_tensors fmt t = Format.(fprintf fmt "@[[%a]]@]" (pp_print_list ~pp_sep:semicolon pp_tensor) t)

  let pp fmt t =
    Format.(fprintf fmt "@[nodes:%d@ inputs:%a@ outputs:%a@]" t.nodes pp_tensors t.inputs pp_tensors t.outputs)

  let compare_tensor a b =
    match Stdlib.compare a.typ b.typ with 0 -> List.compare Int.compare a.shape b.shape | n -> n

  let compare_tensors = List.compare compare_tensor

  let compare a b =
    match Int.compare a.nodes b.nodes with
    | 0 -> ( match compare_tensors a.inputs b.inputs with 0 -> compare_tensors a.outputs b.outputs | n -> n)
    | n -> n
end

module GraphSet = Set.Make (GraphKey)

type callback_state = { label : string; graph_dir : string; mutable observed : GraphSet.t }

let graph_callback_print_something user_data cgraph compute =
  let open Ctypes in
  let nodes = Ggml.C.Functions.graph_n_nodes cgraph in
  let state = Root.get user_data in
  (if compute then
     let key =
       GraphKey.make nodes (Ggml_model_explorer.graph_inputs cgraph) @@ Ggml_model_explorer.graph_outputs cgraph
     in
     if not @@ GraphSet.mem key state.observed then (
       let ptr = raw_address_of_ptr @@ to_voidp cgraph in
       if false then Format.eprintf "@[graph:%#Lx@ key:%a@]@.%!" (Int64.of_nativeint ptr) GraphKey.pp key;
       let n = GraphSet.cardinal state.observed in
       state.observed <- GraphSet.add key state.observed;
       Printf.printf ".[%#LX/%d/%d]." (Int64.of_nativeint ptr) n nodes;
       let json = Filename.concat state.graph_dir @@ Format.sprintf "%s-%d.json" state.label n in
       print_graph ~label:state.label cgraph json));
  if false then Ggml.C.Functions.graph_print cgraph;
  true

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))

let simple fname graph_dir prompt n_predict =
  Ggml.C.Functions_backend.load_all ();
  let open Llama.C.Functions in
  let open Llama.C.Types in
  let model_params = model_default_params () in
  let model = model_load_from_file fname model_params in
  assert (not @@ is_null model);
  let vocab = model_get_vocab model in
  assert (not @@ is_null vocab);
  let n_prompt =
    Int32.neg
    @@ tokenize vocab prompt (Int32.of_int @@ String.length prompt) (from_voidp int32_t null) Int32.zero true true
  in
  let prompt_tokens = CArray.make token @@ Int32.to_int n_prompt in
  let tokens =
    tokenize vocab prompt
      (Int32.of_int @@ String.length prompt)
      (CArray.start prompt_tokens)
      (Int32.of_int @@ CArray.length prompt_tokens)
      true true
  in
  assert (tokens >= Int32.zero);
  let ctx_params = context_default_params () in
  setf ctx_params ContextParams.n_ctx @@ Unsigned.UInt32.of_int @@ (Int32.to_int n_prompt + n_predict - 1);
  setf ctx_params ContextParams.n_batch @@ Unsigned.UInt32.of_int32 n_prompt;
  setf ctx_params ContextParams.no_perf false;

  (* Create the static function pointer *)
  let graph_callback_funptr =
    coerce
      (Foreign.funptr ~runtime_lock:true graph_compute_callback_type)
      graph_compute_callback graph_callback_print_something
  in
  let state_root =
    Option.map
      (fun graph_dir ->
        let state =
          let label = fname |> Filename.basename |> Filename.chop_extension in
          { label; graph_dir; observed = GraphSet.empty }
        in
        let state_root = Root.create state in
        setf ctx_params ContextParams.graph_callback graph_callback_funptr;
        setf ctx_params ContextParams.graph_callback_data state_root;
        state_root)
      graph_dir
  in
  let ctx = init_from_model model ctx_params in
  assert (not @@ is_null ctx);
  let sparams = sampler_chain_default_params () in
  setf sparams SamplerChainParams.no_perf false;
  let smpl = sampler_chain_init sparams in
  assert (not @@ is_null smpl);
  sampler_chain_add smpl @@ sampler_init_greedy ();
  CArray.iter
    (fun id ->
      let buf = CArray.make char 32 in
      let n = token_to_piece vocab id (CArray.start buf) (Int32.of_int @@ CArray.length buf) Int32.zero true in
      assert (n >= Int32.zero);
      let s = Ctypes.string_from_ptr (CArray.start buf) ~length:(Int32.to_int n) in
      print_string s)
    prompt_tokens;
  let batch = batch_get_one (CArray.start prompt_tokens) @@ Int32.of_int @@ CArray.length prompt_tokens in
  let t_main_start = Ggml.C.Functions.time_us () in

  let rec decode_aux ~n_pos ~n_decode ~batch =
    let n_tokens = Int32.to_int @@ getf batch Batch.n_tokens in
    if n_pos + n_tokens < Int32.to_int n_prompt + n_predict then (
      let rc = decode ctx batch in
      assert (rc = Int32.zero);
      let n_pos = n_pos + n_tokens in
      let new_token_id = sampler_sample smpl ctx Int32.minus_one in
      if vocab_is_eog vocab new_token_id then n_decode
      else
        let buf = CArray.make char 128 in
        let n =
          token_to_piece vocab new_token_id (CArray.start buf) (Int32.of_int @@ CArray.length buf) Int32.zero true
        in
        assert (n >= Int32.zero);
        let s = Ctypes.string_from_ptr (CArray.start buf) ~length:(Int32.to_int n) in
        print_string s;
        Stdlib.flush Stdlib.stdout;
        let token_id = allocate int32_t new_token_id in
        let batch = batch_get_one token_id Int32.one in
        decode_aux ~n_pos ~n_decode:(succ n_decode) ~batch)
    else n_decode
  in
  let n_decode = decode_aux ~n_pos:0 ~n_decode:0 ~batch in
  print_newline ();
  let t_main_end = Ggml.C.Functions.time_us () in
  let time = (Int64.to_float @@ Int64.sub t_main_end t_main_start) /. 1_000_000. in
  Printf.eprintf "%s: decoded %d tokens in %.2f s, speed %.2f t/s\n" fname n_decode time (float_of_int n_decode /. time);
  prerr_newline ();
  perf_sampler_print smpl;
  perf_context_print ctx;
  prerr_newline ();
  sampler_free smpl;
  free ctx;
  Option.iter Root.release state_root;
  keep graph_callback_funptr;
  model_free model

let cmd =
  Cmd.v (Cmd.info "simple")
  @@
  let+ model_file = Arg.(required & pos 0 (some file) None & info [] ~docv:"MODEL" ~doc:"Model file")
  and+ prompt = Arg.(value & pos 1 string "Hello my name is" & info [] ~docv:"PROMPT" ~doc:"Prompt")
  and+ predict = Arg.(value & opt int 32 & info [ "n" ] ~doc:"Predict")
  and+ graph_dir =
    Arg.(
      value & opt (some dir) None & info [ "g"; "graph-dir" ] ~docv:"GRAPH_DIR" ~doc:"Directory to write Json graphs")
  in
  simple model_file graph_dir prompt predict

let main () = Cmd.eval cmd
let () = if !Sys.interactive then () else Stdlib.exit @@ main ()
