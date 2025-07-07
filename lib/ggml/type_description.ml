open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "ggml_" ^ name
  let _NS name = "GGML_" ^ name

  let make_enum ?(_NS = _NS) ?_NAME ?(ns = ns) name values =
    let _NAME = match _NAME with Some _NAME -> _NAME | None -> String.uppercase_ascii name in
    let _NAME v = _NS @@ _NAME ^ "_" ^ v in
    enum (ns name) @@ List.map (fun (t, name) -> (t, constant (_NAME name) int64_t)) values

  (** Status codes. Corresponds to C `enum ggml_status`. *)
  let status = make_enum "status" Types.Status.values

  (** Tensor data types. Corresponds to C `enum ggml_type`. *)
  let typ = make_enum "type" Types.Type.values

  (** Precision types. Corresponds to C `enum ggml_prec`. *)
  let prec = make_enum "prec" Types.Prec.values

  (** Model file types. Corresponds to C `enum ggml_ftype`. *)
  let ftype = make_enum "ftype" Types.Ftype.values

  (** Available tensor operations. Corresponds to C `enum ggml_op`. *)
  let op = make_enum "op" Types.Op.values

  (** Unary operations. Corresponds to C `enum ggml_unary_op`. *)
  let unary_op = make_enum "unary_op" Types.UnaryOp.values

  (** Gated linear unit operations. Corresponds to C `enum ggml_glu_op`. *)
  let glu_op = make_enum "glu_op" Types.GluOp.values

  (** Object types. Corresponds to C `enum ggml_object_type`. *)
  let object_type = make_enum "object_type" Types.ObjectType.values

  (** Log levels. Corresponds to C `enum ggml_log_level`. *)
  let log_level = make_enum "log_level" Types.LogLevel.values

  (** Tensor flags. Corresponds to C `enum ggml_tensor_flag`. *)
  let tensor_flag = make_enum "tensor_flag" Types.TensorFlag.values

  (** Pooling operations. Corresponds to C `enum ggml_op_pool`. *)
  let op_pool = make_enum "op_pool" Types.OpPool.values

  (** Sort order. Corresponds to C `enum ggml_sort_order`. *)
  let sort_order = make_enum "sort_order" Types.SortOrder.values

  (** Scale mode for interpolation. Corresponds to C `enum ggml_scale_mode`. *)
  let scale_mode = make_enum "scale_mode" Types.ScaleMode.values

  (** Opaque object structure. Corresponds to C `struct ggml_object`. *)
  let _object' : [ `Object ] structure typ = structure (ns "object")

  let object' = ptr _object'

  (** Opaque context structure. Corresponds to C `struct ggml_context`. *)
  let _context : [ `Context ] structure typ = structure (ns "context")

  let context = ptr _context

  (** Opaque computation graph structure. Corresponds to C `struct ggml_cgraph`. *)
  let cgraph_struct : [ `Cgraph ] structure typ = structure (ns "cgraph")

  let cgraph = ptr cgraph_struct
  let tensor_struct : [ `Tensor ] structure typ = structure (ns "tensor")
  let tensor = ptr tensor_struct
  let const_tensor = ptr @@ const tensor_struct

  (** Opaque backend scheduler structure. Corresponds to C `struct ggml_backend_sched`. *)
  let _backend_sched : [ `BackendSched ] structure typ = structure (ns "backend_sched")

  let backend_sched_t = ptr _backend_sched

  (* Opaque struct types *)

  (** Optimizer parameters. Corresponds to C `struct ggml_opt_params`. *)
  let opt_params_struct : [ `OptParams ] structure typ = structure (ns "opt_params")

  let opt_params_t = ptr opt_params_struct (* Added for consistency, though struct itself is often used directly *)

  (** Opaque optimizer context structure. Corresponds to C `struct ggml_opt_context`. *)
  let _opt_context : [ `OptContext ] structure typ = structure (ns "opt_context")

  let opt_context = ptr _opt_context

  (** Opaque scratch buffer structure. Corresponds to C `struct ggml_scratch`. *)
  let scratch : [ `Scratch ] structure typ = structure (ns "scratch")

  (** Opaque type traits structure. Corresponds to C `struct ggml_type_traits`. *)
  let type_traits : [ `TypeTraits ] structure typ = structure (ns "type_traits")

  (** Opaque threadpool structure. Corresponds to C `struct ggml_threadpool`. *)
  let threadpool : [ `ThreadPool ] structure typ = structure (ns "threadpool")

  (* Typedefs *)

  (** IEEE 754-2008 half-precision float16. Corresponds to C `ggml_fp16_t`. *)
  let fp16_t = typedef uint16_t (ns "fp16_t")

  (** Google Brain half-precision bfloat16. Corresponds to C `ggml_bf16_t`. *)
  let bf16_t = typedef uint16_t (ns "bf16_t")
  (* C struct { uint16_t bits; } - treat as uint16_t for binding *)

  (** GUID type (16 bytes). Corresponds to C `ggml_guid`. *)
  let guid = typedef (array 16 uint8_t) (ns "guid")

  (** Pointer to a GUID. Corresponds to C `ggml_guid_t`. *)
  let guid_t = typedef (ptr guid) (ns "guid_t")

  (* Function pointer types *)

  (** Abort callback. If not NULL, called before ggml computation. If it returns true, the computation is aborted.
      Corresponds to C `ggml_abort_callback`. *)
  let abort_callback = static_funptr (ptr void @-> returning bool)

  (** Function pointer for converting float to a specific type. Corresponds to C `ggml_from_float_t`. *)
  let from_float_t = static_funptr (ptr float @-> ptr void @-> int64_t @-> returning void)

  (** Function pointer for getting optimizer parameters. Corresponds to C `ggml_opt_get_optimizer_params`. *)
  let opt_get_optimizer_params_fn_t = static_funptr (ptr void @-> returning (ptr void))
  (* Return type will be OptOptimizerParams.t *)

  module CPU = struct
    (** Computation plan structure. *)
    module Cplan = struct
      type t

      let t : t structure typ = structure (ns "cplan")

      (** Size of work buffer, calculated by `ggml_graph_plan()`. *)
      let work_size = field t "work_size" size_t

      (** Work buffer, to be allocated by caller before calling `ggml_graph_compute()`. *)
      let work_data = field t "work_data" (ptr uint8_t)

      (** Number of threads to use for computation. *)
      let n_threads = field t "n_threads" int

      (** Optional threadpool instance. *)
      let threadpool = field t "threadpool" (ptr threadpool)

      (** Abort callback. Computation aborts if it returns true. *)
      let abort_callback = field t "abort_callback" abort_callback

      (** Data pointer passed to the abort callback. *)
      let abort_callback_data = field t "abort_callback_data" (ptr void)

      let () = seal t
    end

    (** NUMA strategy. Corresponds to C `enum ggml_numa_strategy`. *)
    let numa_strategy = make_enum "numa_strategy" Types.NumaStrategy.values

    (** Function pointer for vector dot product *)
    let vec_dot_t =
      static_funptr
        (int @-> ptr float @-> size_t @-> ptr void @-> size_t @-> ptr void @-> size_t @-> int @-> returning void)

    (** CPU-specific type traits structure. Corresponds to C `struct ggml_type_traits_cpu`. *)
    module TypeTraitsCpu = struct
      type t

      let t : t structure typ = structure (ns "type_traits_cpu")

      (** Function pointer for converting float to this type. *)
      let from_float = field t "from_float" from_float_t

      (** Function pointer for vector dot product. *)
      let vec_dot = field t "vec_dot" vec_dot_t

      (** Preferred ggml type for vector dot product. *)
      let vec_dot_type = field t "vec_dot_type" typ

      (** Number of rows to process simultaneously. *)
      let nrows = field t "nrows" int64_t

      let () = seal t
    end
  end

  (** Log callback function pointer. Corresponds to C `ggml_log_callback`. *)
  let log_callback = static_funptr (log_level @-> string @-> ptr void @-> returning void)
  (* string for const char* *)

  (** Thread task function pointer. Corresponds to C `ggml_thread_task`. *)
  let thread_task = static_funptr (ptr void @-> int @-> returning void)

  (** Computation graph evaluation callback function pointer. Corresponds to C `ggml_cgraph_eval_callback`. *)
  let cgraph_eval_callback = static_funptr (ptr cgraph @-> ptr void @-> returning bool)

  (** Initialization parameters structure. Corresponds to C `struct ggml_init_params`. *)
  module InitParams = struct
    type t

    let t : t structure typ = structure (ns "init_params")

    (** Memory pool size in bytes. *)
    let mem_size = field t "mem_size" size_t

    (** Memory buffer. If NULL, memory will be allocated internally. *)
    let mem_buffer = field t "mem_buffer" @@ ptr void

    (** Don't allocate memory for tensor data. *)
    let no_alloc = field t "no_alloc" bool

    let () = seal t
  end

  (** n-dimensional tensor structure. Corresponds to C `struct ggml_tensor`. *)
  module Tensor = struct
    open Ggml_const.C.Types

    let t = tensor_struct

    (** Tensor data type *)
    let typ_ = field t "type" typ

    (** Number of elements in each dimension *)
    let ne = field t "ne" (array max_dims int64_t)

    (** Stride in bytes for each dimension *)
    let nb = field t "nb" (array max_dims size_t)

    (** Operation that produced this tensor *)
    let op = field t "op" op

    (** Operation parameters (allocated as int32_t for alignment) *)
    let op_params = field t "op_params" (array (max_op_params / 4) int32_t)

    (** Tensor flags (e.g., input, output, parameter) *)
    let flags = field t "flags" int32_t

    (** Source tensors for this operation *)
    let src = field t "src" (array max_src (ptr t))

    (** Source tensor for view operations *)
    let view_src = field t "view_src" (ptr t)

    (** Offset within the source tensor for view operations *)
    let view_offs = field t "view_offs" size_t

    (** Pointer to the tensor data *)
    let data = field t "data" (ptr void)

    (** Tensor name *)
    let name = field t "name" (array max_name char)

    (** Extra data (e.g., for backend-specific information) *)
    let extra = field t "extra" (ptr void)

    (** Padding for alignment *)
    let padding = field t "padding" (array 8 char)

    let () = seal t
  end

  let guid = array 16 uint8_t

  (** Custom unary operation function pointer type. Corresponds to C `ggml_custom1_op_t`. *)
  let custom1_op_t = static_funptr (tensor @-> const_tensor @-> int @-> int @-> ptr void @-> returning void)

  (** Custom binary operation function pointer type. Corresponds to C `ggml_custom2_op_t`. *)
  let custom2_op_t =
    static_funptr (tensor @-> const_tensor @-> const_tensor @-> int @-> int @-> ptr void @-> returning void)

  (** Custom ternary operation function pointer type. Corresponds to C `ggml_custom3_op_t`. *)
  let custom3_op_t =
    static_funptr
      (tensor @-> const_tensor @-> const_tensor @-> const_tensor @-> int @-> int @-> ptr void @-> returning void)

  (** Custom operation function pointer type (for ggml_custom). Corresponds to C `ggml_custom_op_t`. *)
  let custom_op_t = static_funptr (tensor @-> int @-> int @-> ptr void @-> returning void)

  module Backend = struct
    (** Opaque backend structure *)
    let backend : [ `Backend ] structure typ = structure (ns "backend")

    let backend_t = ptr backend
    let ns name = ns @@ "backend_" ^ name
    let _NS name = _NS @@ "BACKEND_" ^ name

    (** Backend buffer usage status *)
    let status = make_enum "buffer_usage" ~ns ~_NS Types.Backend.BufferUsage.values

    (** Backend device type *)
    let dev_type = make_enum "dev_type" ~ns ~_NS ~_NAME:"DEVICE_TYPE" Types.Backend.DevType.values

    (* Opaque struct types for backend components *)

    (** Opaque backend buffer type structure *)
    let buffer_type_struct : [ `BufferType ] structure typ = structure (ns "buffer_type")

    let buffer_type_t = ptr buffer_type_struct

    (** Opaque backend buffer structure *)
    let buffer_struct : [ `Buffer ] structure typ = structure (ns "buffer")

    let buffer_t = ptr buffer_struct

    (** Opaque backend event structure *)
    let event_struct : [ `Event ] structure typ = structure (ns "event")

    let event_t = ptr event_struct

    (** Opaque backend graph plan type *)
    let graph_plan_t = ptr void

    (** Opaque backend registry structure *)
    let reg_struct : [ `Reg ] structure typ = structure (ns "reg")

    let reg_t = ptr reg_struct

    (** Opaque backend device structure *)
    let dev_struct : [ `Device ] structure typ = structure (ns "device")

    let dev_t = ptr dev_struct

    (** functionality supported by the device *)
    module DevCaps = struct
      type t

      let t : t structure typ = structure (ns "dev_caps")

      (** Supports asynchronous operations *)
      let async = field t "async" bool

      (** Supports pinned host buffers *)
      let host_buffer = field t "host_buffer" bool

      (** Supports creating buffers from host pointers *)
      let buffer_from_host_ptr = field t "buffer_from_host_ptr" bool

      (** Supports event synchronization *)
      let events = field t "events" bool

      let () = seal t
    end

    (** Device properties structure. Mirrors C `ggml_backend_dev_props`. *)
    module DevProps = struct
      type t

      let t : t structure typ = structure (ns "dev_props")

      (** Device name. *)
      let name = field t "name" string

      (** Device description. *)
      let description = field t "description" string

      (** Free memory on the device in bytes. *)
      let memory_free = field t "memory_free" size_t

      (** Total memory on the device in bytes. *)
      let memory_total = field t "memory_total" size_t

      (** Device type (CPU, GPU, ACCEL). *)
      let type_ = field t "type" dev_type

      (** Device capabilities. *)
      let caps = field t "caps" DevCaps.t

      let () = seal t
    end

    (** Backend feature structure. *)
    module BackendFeature = struct
      type t

      let t : t structure typ = structure (ns "feature")

      (** Feature name. *)
      let name = field t "name" string

      (** Feature value (as string). *)
      let value = field t "value" string

      let () = seal t
    end

    (** Structure for copying graphs between backends. *)
    module GraphCopy = struct
      type t

      let t : t structure typ = structure (ns "graph_copy")

      (** Buffer containing the copied graph data. *)
      let buffer = field t "buffer" buffer_t

      (** Context containing allocated tensors of the copied graph. *)
      let ctx_allocated = field t "ctx_allocated" context

      (** Context containing unallocated tensors of the copied graph. *)
      let ctx_unallocated = field t "ctx_unallocated" context

      (** The copied computation graph. *)
      let graph = field t "graph" cgraph

      let () = seal t
    end

    (** Split buffer type for tensor parallelism. *)
    let split_buffer_type_t = static_funptr (int @-> ptr float @-> returning buffer_type_t)

    (** Set the number of threads for the backend. *)
    let set_n_threads_t = static_funptr (backend_t @-> int @-> returning void)

    (** Get additional buffer types provided by the device (returns a NULL-terminated array). *)
    let dev_get_extra_bufts_t = static_funptr (dev_t @-> returning (ptr buffer_type_t))
    (* Returns ptr to buffer_type_t *)

    (** Set the abort callback for the backend. *)
    let set_abort_callback_t = static_funptr (backend_t @-> abort_callback @-> ptr void @-> returning void)

    (** Get features provided by the backend registry. *)
    let get_features_t = static_funptr (reg_t @-> returning (ptr BackendFeature.t))

    (** Backend evaluation callback function pointer *)
    let eval_callback = static_funptr (int @-> tensor @-> tensor @-> ptr void @-> returning bool)

    (** Evaluation callback for the scheduler. *)
    let sched_eval_callback = static_funptr (tensor @-> bool @-> ptr void @-> returning bool)
  end

  module GGUF = struct
    let ns name = "gguf_" ^ name
    let _NS name = "GGUF_" ^ name
    let make_enum name values = make_enum ~_NS ~ns name values
    let typ = make_enum "type" Types.GGUF.Type.values

    (* Opaque type for GGUF context *)
    let context_struct : [ `gguf_context ] structure typ = structure (ns "context")
    let context_t = ptr context_struct

    (* GGUF initialization parameters *)
    module InitParams = struct
      type t

      let t : t structure typ = structure (ns "init_params")

      (** Don't allocate memory for tensor data. *)
      let no_alloc = field t "no_alloc" bool

      (* The C type is 'struct ggml_context ** ctx'. We use the existing ggml context alias 'context' (ptr _context). *)

      (** If not NULL, create a ggml_context and allocate the tensor data in it. *)
      let ctx = field t "ctx" (ptr context)
      (* ptr context = ptr (ptr ggml_context) = ggml_context ** *)

      let () = seal t
    end
  end

  module Opt = struct
    let ns name = ns @@ "opt_" ^ name
    let _NS name = _NS @@ "OPT_" ^ name
    let make_enum name values = make_enum ~_NS ~ns name values

    (** Optimization build types. Corresponds to C `enum ggml_opt_build_type`. *)
    let build_type = make_enum "build_type" Types.Opt.BuildType.values

    (** Loss function types. Corresponds to C `enum ggml_opt_loss_type`. *)
    let loss_type = make_enum "loss_type" Types.Opt.LossType.values

    (** Opaque dataset structure for optimization. Corresponds to C `struct ggml_opt_dataset`. *)
    let _opt_dataset : [ `OptDataset ] structure typ = structure (ns "dataset")

    let opt_dataset_t = ptr _opt_dataset

    (** Opaque optimization result structure. Corresponds to C `struct ggml_opt_result`. *)
    let _opt_result : [ `OptResult ] structure typ = structure (ns "result")

    let opt_result_t = ptr _opt_result
    let opt_result_opt_t = ptr_opt _opt_result (* For functions that can take NULL *)

    (** Optimizer parameters. Corresponds to C `struct ggml_opt_optimizer_params`. *)
    module OptimizerParams = struct
      type t

      let t : t structure typ = structure (ns "optimizer_params")
      let adamw_alpha = field t "adamw.alpha" float
      let adamw_beta1 = field t "adamw.beta1" float
      let adamw_beta2 = field t "adamw.beta2" float
      let adamw_eps = field t "adamw.eps" float
      let adamw_wd = field t "adamw.wd" float
      let () = seal t
    end

    let get_optimizer_params = static_funptr (ptr void @-> returning (* OptimizerParams.t *) void)

    (** Parameters for initializing an optimization context. Corresponds to C `struct ggml_opt_params`. *)
    module Params = struct
      type t

      let t : t structure typ = structure (ns "params")
      let backend_sched = field t "backend_sched" backend_sched_t
      let ctx_compute = field t "ctx_compute" context
      let inputs = field t "inputs" tensor
      let outputs = field t "outputs" tensor
      let loss_type = field t "loss_type" loss_type
      let build_type = field t "build_type" build_type
      let opt_period = field t "opt_period" int32_t

      (* let get_opt_pars = field t "get_opt_pars" @@ static_funptr (ptr void @-> returning OptimizerParams.t) *)
      let get_opt_pars = field t "get_opt_pars" get_optimizer_params
      let get_opt_pars_ud = field t "get_opt_pars_ud" @@ ptr void
      let () = seal t
    end
  end
end
