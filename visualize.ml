open Cmdliner
open Cmdliner.Term.Syntax

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))

let model_file =
  let doc = "Model file to visualize" in
  Arg.(required & pos 0 (some file) None & info [] ~docv:"MODEL" ~doc)

let out_file =
  let doc = "Json file to write" in
  Arg.(required & pos 1 (some string) None & info [] ~docv:"OUT" ~doc)

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

let gpt2 model_file out_file context past tokens =
  let open Gpt_2.C in
  let open Ctypes in
  let model = make ~finalise:(fun m -> Functions.model_uninit @@ addr m) Types.model_t in
  let model' = addr model in
  let rc = Functions.model_init model' model_file context 0 in
  assert (rc = 0);
  let graph = Functions.model_graph model' past tokens in
  print_graph ~label:"gpt2" graph out_file;
  keep (model, graph)

let magika model_file out_file =
  let open Magika.C in
  let open Ctypes in
  let model = make ~finalise:(fun m -> Functions.model_uninit @@ addr m) Types.model_t in
  let rc = Functions.model_init (addr model) model_file in
  assert (rc = 0);
  let graph = Functions.model_graph @@ addr model in
  print_graph ~label:"magika" graph out_file;
  keep (model, graph)

let yolo model_file out_file =
  let open Yolo.C in
  let open Ctypes in
  let model = make ~finalise:(fun m -> Functions.model_uninit @@ addr m) Types.model_t in
  let rc = Functions.model_init (addr model) model_file in
  assert (rc = 0);
  let graph = Functions.model_graph @@ addr model in
  print_graph ~label:"yolo" graph out_file;
  keep (model, graph)

let cmd_simple label cmd =
  let doc = "Visualize " ^ label ^ " graph" in
  Cmd.v (Cmd.info label ~doc)
  @@
  let+ model_file = model_file and+ out_file = out_file in
  cmd model_file out_file

let cmd_gpt2 =
  let label = "gpt2" in
  let doc = "Visualize " ^ label ^ " graph" in
  Cmd.v (Cmd.info label ~doc)
  @@
  let+ model_file = model_file
  and+ out_file = out_file
  and+ context = Arg.(required & opt (some int) (Some 1024) & info [ "c"; "context-size" ])
  and+ past = Arg.(required & opt (some int) (Some 0) & info [ "p"; "past" ])
  and+ tokens = Arg.(required & opt (some int) (Some 768) & info [ "t"; "tokens" ]) in
  gpt2 model_file out_file context past tokens

let cmd_yolo = cmd_simple "yolo" yolo
let cmd_magika = cmd_simple "magika" magika

let cmd =
  let doc = "Visualize ggml graphs" in
  let info = Cmd.info "visualize" ~doc in
  Cmd.group info [ cmd_gpt2; cmd_magika; cmd_yolo ]

let main () = Cmd.eval cmd
let () = if !Sys.interactive then () else exit (main ())
