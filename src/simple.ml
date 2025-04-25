open Ctypes
open Cmdliner
open Cmdliner.Term.Syntax

type callback_state = { fname : string }

let graph_callback_print_something user_data cgraph compute =
  let nodes = Ggml.C.Functions.graph_n_nodes cgraph in
  let state = Ctypes.Root.get user_data in
  ignore (compute,state,nodes);
  if false then Ggml.C.Functions.graph_print cgraph;
  true

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))

let simple fname prompt n_predict =
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
  let state = { fname } in
  let state_root = Root.create state in
  setf ctx_params ContextParams.graph_callback graph_callback_funptr;
  setf ctx_params ContextParams.graph_callback_data state_root;

  (* Revert to setting the funptr if direct fails *)
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
  Root.release state_root;
  keep (graph_callback_funptr, state_root);
  model_free model

let cmd =
  Cmd.v (Cmd.info "simple")
  @@
  let+ model_file = Arg.(required & pos 0 (some file) None & info [] ~docv:"MODEL" ~doc:"Model file")
  and+ prompt = Arg.(value & pos 1 string "Hello my name is" & info [] ~docv:"PROMPT" ~doc:"Prompt")
  and+ predict = Arg.(value & opt int 32 & info [ "n" ] ~doc:"Predict") in
  simple model_file prompt predict

let main () = Cmd.eval cmd
let () = if !Sys.interactive then () else Stdlib.exit @@ main ()
