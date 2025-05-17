open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated
  open Types_generated.Opt

  (** [dataset_init type_data type_label ne_datapoint ne_label ndata ndata_shard] initializes a dataset for
      optimization.
      - [type_data] The type for the internal data tensor.
      - [type_label] The type for the internal labels tensor.
      - [ne_datapoint] Number of elements per datapoint.
      - [ne_label] Number of elements per label.
      - [ndata] Total number of datapoints/labels.
      - [ndata_shard] Number of datapoints/labels per shard (unit for shuffling/copying).
      - returns The initialized dataset. *)
  let dataset_init =
    foreign (ns "dataset_init")
      (typ @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning Opt.opt_dataset_t)

  (** [dataset_free dataset] frees the memory associated with a dataset.
      - [dataset] The dataset to free. *)
  let dataset_free = foreign (ns "dataset_free") (Opt.opt_dataset_t @-> returning void)

  (** [dataset_ndata dataset] returns the total number of datapoints/labels in the dataset.
      - [dataset] The dataset.
      - returns The total number of datapoints/labels. *)
  let dataset_ndata = foreign (ns "dataset_ndata") (Opt.opt_dataset_t @-> returning int64_t)

  (** [dataset_data dataset] returns the underlying tensor that stores the data. Shape = [ne_datapoint, ndata].
      - [dataset] The dataset.
      - returns The data tensor. *)
  let dataset_data = foreign (ns "dataset_data") (Opt.opt_dataset_t @-> returning tensor)

  (** [dataset_labels dataset] returns the underlying tensor that stores the labels. Shape = [ne_label, ndata].
      - [dataset] The dataset.
      - returns The labels tensor. *)
  let dataset_labels = foreign (ns "dataset_labels") (Opt.opt_dataset_t @-> returning tensor)

  (** [dataset_get_batch dataset data_batch labels_batch ibatch] copies a batch of data and labels from the dataset to
      the provided tensors.
      - [dataset] The dataset.
      - [data_batch] Tensor to store the data batch (shape = [ne_datapoint, ndata_batch]).
      - [labels_batch] Tensor to store the labels batch (shape = [ne_label, ndata_batch]).
      - [ibatch] The batch index. *)
  let dataset_get_batch =
    foreign (ns "dataset_get_batch") (Opt.opt_dataset_t @-> tensor @-> tensor @-> int64_t @-> returning void)

  (** [dataset_get_batch_host dataset data_batch_ptr nb_data_batch labels_batch_ptr ibatch] copies a batch of data and
      labels from the dataset to host memory.
      - [dataset] The dataset.
      - [data_batch_ptr] Pointer to host memory for the data batch.
      - [nb_data_batch] Size in bytes of the data batch buffer.
      - [labels_batch_ptr] Pointer to host memory for the labels batch.
      - [ibatch] The batch index. *)
  let dataset_get_batch_host =
    foreign (ns "dataset_get_batch_host")
      (Opt.opt_dataset_t @-> ptr void @-> size_t @-> ptr void @-> int64_t @-> returning void)

  (** [get_default_optimizer_params userdata] returns the default optimizer parameters (constant, hard-coded values).
      userdata is not used.
      - [userdata] User data (not used).
      - returns The default optimizer parameters. *)
  let get_default_optimizer_params =
    foreign (ns "get_default_optimizer_params") (ptr void @-> returning Opt.OptimizerParams.t)

  (** [get_constant_optimizer_params userdata] casts userdata to ggml_opt_optimizer_params and returns it.
      - [userdata] Pointer to ggml_opt_optimizer_params.
      - returns The optimizer parameters. *)
  let get_constant_optimizer_params =
    foreign (ns "get_constant_optimizer_params") (ptr void @-> returning Opt.OptimizerParams.t)

  (** [default_params backend_sched loss_type] gets parameters for an optimization context with defaults set where
      possible.
      - [backend_sched] Defines which backends are used to construct the compute graphs.
      - [loss_type] The type of loss function to use.
      - returns The default optimization parameters. *)
  let default_params = foreign (ns "default_params") (backend_sched_t @-> Opt.loss_type @-> returning Opt.Params.t)

  (** [init params] initializes a new optimization context.
      - [params] The optimization parameters.
      - returns The initialized optimization context. *)
  let init = foreign (ns "init") (Opt.Params.t @-> returning opt_context)

  (** [free opt_ctx] frees the memory associated with an optimization context.
      - [opt_ctx] The optimization context to free. *)
  let free = foreign (ns "free") (opt_context @-> returning void)

  (** [reset opt_ctx optimizer] resets an optimization context.
      - [opt_ctx] The optimization context.
      - [optimizer] Whether to reset the optimizer state. *)
  let reset = foreign (ns "reset") (opt_context @-> bool @-> returning void)

  (** [inputs opt_ctx] returns the forward graph input tensor. If not using static graphs these pointers become invalid
      with the next call to ggml_opt_alloc.
      - [opt_ctx] The optimization context.
      - returns The input tensor. *)
  let inputs = foreign (ns "inputs") (opt_context @-> returning tensor)

  (** [outputs opt_ctx] returns the forward graph output tensor. If not using static graphs these pointers become
      invalid with the next call to ggml_opt_alloc.
      - [opt_ctx] The optimization context.
      - returns The output tensor. *)
  let outputs = foreign (ns "outputs") (opt_context @-> returning tensor)

  (** [labels opt_ctx] returns the labels tensor to compare outputs against.
      - [opt_ctx] The optimization context.
      - returns The labels tensor. *)
  let labels = foreign (ns "labels") (opt_context @-> returning tensor)

  (** [loss opt_ctx] returns the loss tensor.
      - [opt_ctx] The optimization context.
      - returns The loss tensor. *)
  let loss = foreign (ns "loss") (opt_context @-> returning tensor)

  (** [pred opt_ctx] returns the predictions made by outputs.
      - [opt_ctx] The optimization context.
      - returns The prediction tensor. *)
  let pred = foreign (ns "pred") (opt_context @-> returning tensor)

  (** [ncorrect opt_ctx] returns the number of matching predictions between outputs and labels.
      - [opt_ctx] The optimization context.
      - returns The ncorrect tensor. *)
  let ncorrect = foreign (ns "ncorrect") (opt_context @-> returning tensor)

  (** [grad_acc opt_ctx node] returns the gradient accumulator for a node from the forward graph.
      - [opt_ctx] The optimization context.
      - [node] The tensor node from the forward graph.
      - returns The gradient accumulation tensor for the node. *)
  let grad_acc = foreign (ns "grad_acc") (opt_context @-> tensor @-> returning tensor)

  (** [result_init ()] initializes an optimization result structure.
      - returns The initialized optimization result. *)
  let result_init = foreign (ns "result_init") (void @-> returning Opt.opt_result_t)

  (** [result_free result] frees the memory associated with an optimization result.
      - [result] The optimization result to free. *)
  let result_free = foreign (ns "result_free") (Opt.opt_result_t @-> returning void)

  (** [result_reset result] resets an optimization result.
      - [result] The optimization result to reset. *)
  let result_reset = foreign (ns "result_reset") (Opt.opt_result_t @-> returning void)

  (** [prepare_alloc opt_ctx ctx_compute gf inputs outputs] prepares for graph allocation if not using static graphs.
      This function must be called prior to `opt_alloc`.
      - [opt_ctx] The optimization context.
      - [ctx_compute] Context for temporary tensors.
      - [gf] The forward graph.
      - [inputs] Input tensor for the forward graph.
      - [outputs] Output tensor for the forward graph. *)
  let prepare_alloc =
    foreign (ns "prepare_alloc") (opt_context @-> context @-> cgraph @-> tensor @-> tensor @-> returning void)

  (** [alloc opt_ctx backward] allocates the next graph for evaluation, either forward or forward + backward. Must be
      called exactly once prior to calling `opt_eval`.
      - [opt_ctx] The optimization context.
      - [backward] Whether to allocate for a backward pass. *)
  let alloc = foreign (ns "alloc") (opt_context @-> bool @-> returning void)

  (** [eval opt_ctx result] performs a forward pass, increments result if not NULL, and performs a backward pass if
      allocated.
      - [opt_ctx] The optimization context.
      - [result] Optional optimization result structure to update. Can be NULL. *)
  let eval = foreign (ns "eval") (opt_context @-> Opt.opt_result_opt_t @-> returning void)

  (* let get_optimizer_params = Foreign.funptr Ctypes.(ptr void @-> returning Opt.OptimizerParams.t) *)

  (** [fit backend_sched ctx_compute inputs outputs dataset loss_type get_opt_pars userdata n_epochs n_iter_max] fits a
      model defined by inputs and outputs to a dataset.
      - [backend_sched] Backend scheduler for constructing compute graphs.
      - [ctx_compute] Context with temporarily allocated tensors to calculate outputs.
      - [inputs] Input tensor with shape [ne_datapoint, ndata_batch].
      - [outputs] Output tensor, must have shape [ne_label, ndata_batch] if labels are used.
      - [dataset] Dataset with data and optionally also labels.
      - [loss_type] Loss function to minimize.
      - [get_opt_pars] Callback to get optimizer parameters; userdata is a pointer to epoch (int64_t).
      - [userdata] User data for the `get_opt_pars` callback.
      - [n_epochs] Number of epochs to train for.
      - [n_iter_max] Maximum number of iterations per epoch. *)
  let fit =
    foreign (ns "fit")
      (backend_sched_t @-> context @-> tensor @-> tensor @-> Opt.opt_dataset_t @-> Opt.loss_type
     @-> get_optimizer_params @-> ptr void @-> int64_t @-> int64_t @-> returning void)
end
