open Ctypes

module Types (F : Ctypes.TYPE) = struct
  open F

  let ns name = "ggml_" ^ name
  let _NS name = "GGML_" ^ name

  let make_enum name values =
    let _NAME v = _NS @@ String.uppercase_ascii name ^ "_" ^ v in
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
  let numa_strategy = make_enum "numa_strategy" Types.NumaStrategy.values
  let _object' : [ `Object ] structure typ = structure (ns "object")
  let object' = ptr _object'
  let _context : [ `Context ] structure typ = structure (ns "context")
  let context = ptr _context
  let _cgraph : [ `Cgraph ] structure typ = structure (ns "cgraph")
  let cgraph = ptr _cgraph
  let backend_buffer : [ `BackendBuffer ] structure typ = structure (ns "backend_buffer")

  (* Opaque struct types *)
  let opt_params : [ `OptParams ] structure typ = structure (ns "opt_params")
  let opt_context : [ `OptContext ] structure typ = structure (ns "opt_context")
  let scratch : [ `Scratch ] structure typ = structure (ns "scratch")
  let type_traits : [ `TypeTraits ] structure typ = structure (ns "type_traits")
  let threadpool : [ `ThreadPool ] structure typ = structure (ns "threadpool")
  let backend : [ `Backend ] structure typ = structure (ns "backend")
  let backend_t = ptr backend
  let backend_reg : [ `BackendReg ] structure typ = structure (ns "backend_reg")
  let backend_reg_t = ptr backend_reg

  (* Typedefs *)
  let fp16_t = typedef uint16_t (ns "fp16_t")
  let bf16_t = typedef uint16_t (ns "bf16_t") (* C struct { uint16_t bits; } - treat as uint16_t for binding *)
  let guid = typedef (array 16 uint8_t) (ns "guid")
  let guid_t = typedef (ptr guid) (ns "guid_t")

  (* Function pointer types *)
  let abort_callback = static_funptr (ptr void @-> returning bool)
  let from_float_t = static_funptr (ptr float @-> ptr void @-> int64_t @-> returning void)

  let vec_dot_t =
    static_funptr
      (int @-> ptr float @-> size_t @-> ptr void @-> size_t @-> ptr void @-> size_t @-> int @-> returning void)

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

  let cplan = Cplan.t (* Keep the alias for compatibility if needed *)
  let log_callback = static_funptr (log_level @-> string @-> ptr void @-> returning void) (* string for const char* *)
  let thread_task = static_funptr (ptr void @-> int @-> returning void)
  let cgraph_eval_callback = static_funptr (ptr cgraph @-> ptr void @-> returning bool)

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
    let buffer = field t "buffer" (ptr backend_buffer)
    let ne = field t "ne" (array max_dims int64_t)
    let nb = field t "nb" (array max_dims size_t)
    let op_ = field t "op" op
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
  let guid = array 16 uint8_t
end
