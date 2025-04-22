open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "ggml_" ^ name
  let _NS name = "GGML_" ^ name

  let make_enum ?(_NS = _NS) ?_NAME ?(ns = ns) name values =
    let _NAME = match _NAME with Some _NAME -> _NAME | None -> String.uppercase_ascii name in
    let _NAME v = _NS @@ _NAME ^ "_" ^ v in
    enum (ns name) @@ List.map (fun (t, name) -> (t, constant (_NAME name) int64_t)) values

  let status = make_enum "status" Types.Status.values
  let typ = make_enum "type" Types.Type.values
  let prec = make_enum "prec" Types.Prec.values
  let ftype = make_enum "ftype" Types.Ftype.values
  let op = make_enum "op" Types.Op.values
  let unary_op = make_enum "unary_op" Types.UnaryOp.values
  let object_type = make_enum "object_type" Types.ObjectType.values
  let log_level = make_enum "log_level" Types.LogLevel.values
  let tensor_flag = make_enum "tensor_flag" Types.TensorFlag.values
  let op_pool = make_enum "op_pool" Types.OpPool.values
  let sort_order = make_enum "sort_order" Types.SortOrder.values
  let scale_mode = make_enum "scale_mode" Types.ScaleMode.values
  let _object' : [ `Object ] structure typ = structure (ns "object")
  let object' = ptr _object'
  let _context : [ `Context ] structure typ = structure (ns "context")
  let context = ptr _context
  let _cgraph : [ `Cgraph ] structure typ = structure (ns "cgraph")
  let cgraph = ptr _cgraph

  (* Opaque struct types *)
  let opt_params : [ `OptParams ] structure typ = structure (ns "opt_params")
  let opt_context : [ `OptContext ] structure typ = structure (ns "opt_context")
  let scratch : [ `Scratch ] structure typ = structure (ns "scratch")
  let type_traits : [ `TypeTraits ] structure typ = structure (ns "type_traits")
  let threadpool : [ `ThreadPool ] structure typ = structure (ns "threadpool")

  (* Typedefs *)
  let fp16_t = typedef uint16_t (ns "fp16_t")
  let bf16_t = typedef uint16_t (ns "bf16_t") (* C struct { uint16_t bits; } - treat as uint16_t for binding *)
  let guid = typedef (array 16 uint8_t) (ns "guid")
  let guid_t = typedef (ptr guid) (ns "guid_t")

  (* Function pointer types *)
  let abort_callback = static_funptr (ptr void @-> returning bool)
  let from_float_t = static_funptr (ptr float @-> ptr void @-> int64_t @-> returning void)

  module CPU = struct
    (** Computation plan structure. *)
    module Cplan = struct
      type t

      let t : t structure typ = structure (ns "cplan")
      let work_size = field t "work_size" size_t
      let work_data = field t "work_data" (ptr uint8_t)
      let n_threads = field t "n_threads" int
      let threadpool = field t "threadpool" (ptr threadpool)
      let abort_callback = field t "abort_callback" abort_callback
      let abort_callback_data = field t "abort_callback_data" (ptr void)
      let () = seal t
    end

    let numa_strategy = make_enum "numa_strategy" Types.NumaStrategy.values

    let vec_dot_t =
      static_funptr
        (int @-> ptr float @-> size_t @-> ptr void @-> size_t @-> ptr void @-> size_t @-> int @-> returning void)

    (** CPU-specific type traits structure. *)
    module TypeTraitsCpu = struct
      type t

      let t : t structure typ = structure (ns "type_traits_cpu")
      let from_float = field t "from_float" from_float_t
      let vec_dot = field t "vec_dot" vec_dot_t
      let vec_dot_type = field t "vec_dot_type" typ
      let nrows = field t "nrows" int64_t
      let () = seal t
    end
  end

  let log_callback = static_funptr (log_level @-> string @-> ptr void @-> returning void) (* string for const char* *)
  let thread_task = static_funptr (ptr void @-> int @-> returning void)
  let cgraph_eval_callback = static_funptr (ptr cgraph @-> ptr void @-> returning bool)

  (** Initialization parameters structure. *)
  module InitParams = struct
    type t

    let t : t structure typ = structure (ns "init_params")
    let mem_size = field t "mem_size" size_t
    let mem_buffer = field t "mem_buffer" @@ ptr void
    let no_alloc = field t "no_alloc" bool
    let () = seal t
  end

  (** n-dimensional tensor structure. *)
  module Tensor = struct
    open Ggml_const.C.Types

    type t

    let t : t structure typ = structure (ns "tensor")

    (* Need to declare t before using it recursively in src and view_src *)
    let typ_ = field t "type" typ
    let ne = field t "ne" (array max_dims int64_t)
    let nb = field t "nb" (array max_dims size_t)
    let op = field t "op" op
    let op_params = field t "op_params" (array (max_op_params / 4) int32_t)
    let flags = field t "flags" int32_t
    let src = field t "src" (array max_src (ptr t))
    let view_src = field t "view_src" (ptr t)
    let view_offs = field t "view_offs" size_t
    let data = field t "data" (ptr void)
    let name = field t "name" (array max_name char)
    let extra = field t "extra" (ptr void)
    let padding = field t "padding" (array 8 char)
    let () = seal t
  end

  let tensor = ptr Tensor.t
  let const_tensor = ptr @@ const Tensor.t
  let guid = array 16 uint8_t

  module Backend = struct
    let backend : [ `Backend ] structure typ = structure (ns "backend")
    let backend_t = ptr backend
    let ns name = ns @@ "backend_" ^ name
    let _NS name = _NS @@ "BACKEND_" ^ name
    let status = make_enum "buffer_usage" ~ns ~_NS Types.Backend.BufferUsage.values
    let dev_type = make_enum "dev_type" ~ns ~_NS ~_NAME:"DEVICE_TYPE" Types.Backend.DevType.values

    (* Opaque struct types for backend components *)
    let buffer_type_struct : [ `BufferType ] structure typ = structure (ns "buffer_type")
    let buffer_type_t = ptr buffer_type_struct
    let buffer_struct : [ `Buffer ] structure typ = structure (ns "buffer")
    let buffer_t = ptr buffer_struct
    let event_struct : [ `Event ] structure typ = structure (ns "event")
    let event_t = ptr event_struct
    let graph_plan_t = ptr void
    let reg_struct : [ `Reg ] structure typ = structure (ns "reg")
    let reg_t = ptr reg_struct
    let dev_struct : [ `Device ] structure typ = structure (ns "device")
    let dev_t = ptr dev_struct

    (** functionality supported by the device *)
    module DevCaps = struct
      type t

      let t : t structure typ = structure (ns "dev_caps")
      let async = field t "async" bool
      let host_buffer = field t "host_buffer" bool
      let buffer_from_host_ptr = field t "buffer_from_host_ptr" bool
      let events = field t "events" bool
      let () = seal t
    end

    (** Device properties structure. Mirrors C `ggml_backend_dev_props`. *)
    module DevProps = struct
      type t

      let t : t structure typ = structure (ns "dev_props")
      let name = field t "name" string
      let description = field t "description" string
      let memory_free = field t "memory_free" size_t
      let memory_total = field t "memory_total" size_t
      let type_ = field t "type" dev_type
      let caps = field t "caps" DevCaps.t
      let () = seal t
    end

    (** Backend feature structure. *)
    module BackendFeature = struct
      type t

      let t : t structure typ = structure (ns "feature")
      let name = field t "name" string
      let value = field t "value" string
      let () = seal t
    end

    (** Structure for copying graphs between backends. *)
    module GraphCopy = struct
      type t

      let t : t structure typ = structure (ns "graph_copy")
      let buffer = field t "buffer" buffer_t
      let ctx_allocated = field t "ctx_allocated" context
      let ctx_unallocated = field t "ctx_unallocated" context
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

    let eval_callback = static_funptr (int @-> tensor @-> tensor @-> ptr void @-> returning bool)

    (** Evaluation callback for the scheduler. *)
    let sched_eval_callback = static_funptr (tensor @-> bool @-> ptr void @-> returning bool)
  end

  module GGUF = struct
    let ns name = "gguf_" ^ name
    let _NS name = "GGUF_" ^ name
    let make_enum = make_enum ~_NS ~ns
    let typ = make_enum "type" Types.GGUF.Type.values

    (* Opaque type for GGUF context *)
    let context_struct : [ `gguf_context ] structure typ = structure (ns "context")
    let context_t = ptr context_struct

    (* GGUF initialization parameters *)
    module InitParams = struct
      type t

      let t : t structure typ = structure (ns "init_params")
      let no_alloc = field t "no_alloc" bool

      (* The C type is 'struct ggml_context ** ctx'. We use the existing ggml context alias 'context' (ptr _context). *)
      let ctx = field t "ctx" (ptr context) (* ptr context = ptr (ptr ggml_context) = ggml_context ** *)
      let () = seal t
    end
  end
end
