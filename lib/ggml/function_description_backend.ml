open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated
  open Types_generated.Backend

  (** Backend Buffer Type *)

  (** [buft_name buft] returns the name of the backend buffer type.
      - [buft] The buffer type.
      - returns The name of the buffer type. *)
  let buft_name = foreign (ns "buft_name") (buffer_type_t @-> returning string)

  (** [buft_alloc_buffer buft size] allocates a buffer of the specified size using the buffer type.
      - [buft] The buffer type.
      - [size] The size of the buffer in bytes.
      - returns The allocated buffer. *)
  let buft_alloc_buffer = foreign (ns "buft_alloc_buffer") (buffer_type_t @-> size_t @-> returning buffer_t)

  (** [buft_get_alignment buft] returns the required memory alignment for the buffer type.
      - [buft] The buffer type.
      - returns The alignment in bytes. *)
  let buft_get_alignment = foreign (ns "buft_get_alignment") (buffer_type_t @-> returning size_t)

  (** [buft_get_max_size buft] returns the maximum buffer size supported by the buffer type.
      - [buft] The buffer type.
      - returns The maximum size in bytes, or 0 if unknown. *)
  let buft_get_max_size = foreign (ns "buft_get_max_size") (buffer_type_t @-> returning size_t)

  (** [buft_get_alloc_size buft tensor] returns the size required to allocate the tensor in this buffer type.
      - [buft] The buffer type.
      - [tensor] The tensor.
      - returns The allocation size in bytes. *)
  let buft_get_alloc_size = foreign (ns "buft_get_alloc_size") (buffer_type_t @-> tensor @-> returning size_t)

  (** [buft_is_host buft] checks if the buffer type represents host memory (CPU).
      - [buft] The buffer type.
      - returns True if it's host memory, false otherwise. *)
  let buft_is_host = foreign (ns "buft_is_host") (buffer_type_t @-> returning bool)

  (** [buft_get_device buft] returns the device associated with the buffer type.
      - [buft] The buffer type.
      - returns The backend device. *)
  let buft_get_device = foreign (ns "buft_get_device") (buffer_type_t @-> returning dev_t)

  (** Backend Buffer *)

  (** [buffer_name buffer] returns the name of the backend buffer.
      - [buffer] The buffer.
      - returns The name of the buffer. *)
  let buffer_name = foreign (ns "buffer_name") (buffer_t @-> returning string)

  (** [buffer_free buffer] frees the backend buffer.
      - [buffer] The buffer to free. *)
  let buffer_free = foreign (ns "buffer_free") (buffer_t @-> returning void)

  (** [buffer_get_base buffer] returns a pointer to the base memory of the buffer.
      - [buffer] The buffer.
      - returns A void pointer to the buffer's base memory. Returns NULL for non-CPU memory. *)
  let buffer_get_base = foreign (ns "buffer_get_base") (buffer_t @-> returning (ptr void))

  (** [buffer_get_size buffer] returns the size of the buffer in bytes.
      - [buffer] The buffer.
      - returns The size in bytes. *)
  let buffer_get_size = foreign (ns "buffer_get_size") (buffer_t @-> returning size_t)

  (** [buffer_init_tensor buffer tensor] initializes a tensor with memory from the buffer.
      - [buffer] The buffer.
      - [tensor] The tensor to initialize.
      - returns Status code (`GGML_STATUS_SUCCESS` on success). *)
  let buffer_init_tensor = foreign (ns "buffer_init_tensor") (buffer_t @-> tensor @-> returning status)

  (** [buffer_get_alignment buffer] returns the required memory alignment for the buffer.
      - [buffer] The buffer.
      - returns The alignment in bytes. *)
  let buffer_get_alignment = foreign (ns "buffer_get_alignment") (buffer_t @-> returning size_t)

  (** [buffer_get_max_size buffer] returns the maximum buffer size supported by the buffer.
      - [buffer] The buffer.
      - returns The maximum size in bytes, or 0 if unknown. *)
  let buffer_get_max_size = foreign (ns "buffer_get_max_size") (buffer_t @-> returning size_t)

  (** [buffer_get_alloc_size buffer tensor] returns the size required to allocate the tensor in this buffer.
      - [buffer] The buffer.
      - [tensor] The tensor.
      - returns The allocation size in bytes. *)
  let buffer_get_alloc_size = foreign (ns "buffer_get_alloc_size") (buffer_t @-> tensor @-> returning size_t)

  (** [buffer_clear buffer value] clears the buffer memory with the specified byte value.
      - [buffer] The buffer.
      - [value] The byte value to set. *)
  let buffer_clear = foreign (ns "buffer_clear") (buffer_t @-> uint8_t @-> returning void)

  (** [buffer_is_host buffer] checks if the buffer represents host memory (CPU).
      - [buffer] The buffer.
      - returns True if it's host memory, false otherwise. *)
  let buffer_is_host = foreign (ns "buffer_is_host") (buffer_t @-> returning bool)

  (** [buffer_set_usage buffer usage] sets the usage type for the buffer (e.g., weights, compute).
      - [buffer] The buffer.
      - [usage] The buffer usage type. *)
  let buffer_set_usage = foreign (ns "buffer_set_usage") (buffer_t @-> status @-> returning void)
  (* status is ggml_backend_buffer_usage *)

  (** [buffer_get_usage buffer] gets the usage type of the buffer.
      - [buffer] The buffer.
      - returns The buffer usage type. *)
  let buffer_get_usage = foreign (ns "buffer_get_usage") (buffer_t @-> returning status)

  (** [buffer_get_type buffer] gets the buffer type associated with the buffer.
      - [buffer] The buffer.
      - returns The buffer type. *)
  let buffer_get_type = foreign (ns "buffer_get_type") (buffer_t @-> returning buffer_type_t)

  (** [buffer_reset buffer] resets the buffer's allocation state.
      - [buffer] The buffer. *)
  let buffer_reset = foreign (ns "buffer_reset") (buffer_t @-> returning void)

  (** Tensor Copy *)

  (** [tensor_copy src dst] copies the data from tensor `src` to tensor `dst`. Handles copies between different
      backends.
      - [src] The source tensor.
      - [dst] The destination tensor. *)
  let tensor_copy = foreign (ns "tensor_copy") (tensor @-> tensor @-> returning void)

  (** Backend (Stream) *)

  (** [guid backend] returns the GUID of the backend.
      - [backend] The backend.
      - returns The GUID. *)
  let guid = foreign (ns "guid") (backend_t @-> returning guid_t)

  (** [name backend] returns the name of the backend.
      - [backend] The backend.
      - returns The name. *)
  let name = foreign (ns "name") (backend_t @-> returning string)

  (** [free backend] frees the backend resources.
      - [backend] The backend to free. *)
  let free = foreign (ns "free") (backend_t @-> returning void)

  (** [get_default_buffer_type backend] returns the default buffer type for the backend.
      - [backend] The backend.
      - returns The default buffer type. *)
  let get_default_buffer_type = foreign (ns "get_default_buffer_type") (backend_t @-> returning buffer_type_t)

  (** [alloc_buffer backend size] allocates a buffer using the backend's default buffer type.
      - [backend] The backend.
      - [size] The size of the buffer in bytes.
      - returns The allocated buffer. *)
  let alloc_buffer = foreign (ns "alloc_buffer") (backend_t @-> size_t @-> returning buffer_t)

  (** [get_alignment backend] returns the required memory alignment for the backend.
      - [backend] The backend.
      - returns The alignment in bytes. *)
  let get_alignment = foreign (ns "get_alignment") (backend_t @-> returning size_t)

  (** [get_max_size backend] returns the maximum buffer size supported by the backend.
      - [backend] The backend.
      - returns The maximum size in bytes, or 0 if unknown. *)
  let get_max_size = foreign (ns "get_max_size") (backend_t @-> returning size_t)

  (** [tensor_set_async backend tensor data offset size] asynchronously sets tensor data.
      - [backend] The backend.
      - [tensor] The tensor to modify.
      - [data] Pointer to the source data.
      - [offset] Offset in the tensor's data buffer (bytes).
      - [size] Size of the data to copy (bytes). *)
  let tensor_set_async =
    foreign (ns "tensor_set_async") (backend_t @-> tensor @-> ptr void @-> size_t @-> size_t @-> returning void)

  (** [tensor_get_async backend tensor data offset size] asynchronously gets tensor data.
      - [backend] The backend.
      - [tensor] The tensor to read from.
      - [data] Pointer to the destination buffer.
      - [offset] Offset in the tensor's data buffer (bytes).
      - [size] Size of the data to copy (bytes). *)
  let tensor_get_async =
    foreign (ns "tensor_get_async") (backend_t @-> const_tensor @-> ptr void @-> size_t @-> size_t @-> returning void)

  (** [tensor_set tensor data offset size] synchronously sets tensor data.
      - [tensor] The tensor to modify.
      - [data] Pointer to the source data.
      - [offset] Offset in the tensor's data buffer (bytes).
      - [size] Size of the data to copy (bytes). *)
  let tensor_set = foreign (ns "tensor_set") (tensor @-> ptr void @-> size_t @-> size_t @-> returning void)

  (** [tensor_get tensor data offset size] synchronously gets tensor data.
      - [tensor] The tensor to read from.
      - [data] Pointer to the destination buffer.
      - [offset] Offset in the tensor's data buffer (bytes).
      - [size] Size of the data to copy (bytes). *)
  let tensor_get = foreign (ns "tensor_get") (const_tensor @-> ptr void @-> size_t @-> size_t @-> returning void)

  (** [tensor_memset tensor value offset size] sets a region of the tensor's data to a specific byte value.
      - [tensor] The tensor to modify.
      - [value] The byte value to set.
      - [offset] Offset in the tensor's data buffer (bytes).
      - [size] Size of the region to set (bytes). *)
  let tensor_memset = foreign (ns "tensor_memset") (tensor @-> uint8_t @-> size_t @-> size_t @-> returning void)

  (** [synchronize backend] synchronizes the backend, ensuring all pending operations are complete.
      - [backend] The backend. *)
  let synchronize = foreign (ns "synchronize") (backend_t @-> returning void)

  (** [graph_plan_create backend cgraph] creates a compute plan for the graph on the backend.
      - [backend] The backend.
      - [cgraph] The computation graph.
      - returns The compute plan. *)
  let graph_plan_create = foreign (ns "graph_plan_create") (backend_t @-> cgraph @-> returning graph_plan_t)

  (** [graph_plan_free backend plan] frees the compute plan.
      - [backend] The backend.
      - [plan] The compute plan to free. *)
  let graph_plan_free = foreign (ns "graph_plan_free") (backend_t @-> graph_plan_t @-> returning void)

  (** [graph_plan_compute backend plan] executes the compute plan.
      - [backend] The backend.
      - [plan] The compute plan.
      - returns Status code. *)
  let graph_plan_compute = foreign (ns "graph_plan_compute") (backend_t @-> graph_plan_t @-> returning status)

  (** [graph_compute backend cgraph] computes the graph directly on the backend.
      - [backend] The backend.
      - [cgraph] The computation graph.
      - returns Status code. *)
  let graph_compute = foreign (ns "graph_compute") (backend_t @-> cgraph @-> returning status)

  (** [graph_compute_async backend cgraph] asynchronously computes the graph on the backend.
      - [backend] The backend.
      - [cgraph] The computation graph.
      - returns Status code. *)
  let graph_compute_async = foreign (ns "graph_compute_async") (backend_t @-> cgraph @-> returning status)

  (** [supports_op backend op] checks if the backend supports the operation of the given tensor.
      - [backend] The backend.
      - [op] The tensor representing the operation.
      - returns True if supported, false otherwise. *)
  let supports_op = foreign (ns "supports_op") (backend_t @-> const_tensor @-> returning bool)

  (** [supports_buft backend buft] checks if the backend supports the given buffer type.
      - [backend] The backend.
      - [buft] The buffer type.
      - returns True if supported, false otherwise. *)
  let supports_buft = foreign (ns "supports_buft") (backend_t @-> buffer_type_t @-> returning bool)

  (** [offload_op backend op] checks if the operation should be offloaded to this backend.
      - [backend] The backend.
      - [op] The tensor representing the operation.
      - returns True if it should be offloaded, false otherwise. *)
  let offload_op = foreign (ns "offload_op") (backend_t @-> const_tensor @-> returning bool)

  (** [tensor_copy_async backend_src backend_dst src dst] asynchronously copies a tensor between backends.
      - [backend_src] The source backend.
      - [backend_dst] The destination backend.
      - [src] The source tensor.
      - [dst] The destination tensor. *)
  let tensor_copy_async =
    foreign (ns "tensor_copy_async") (backend_t @-> backend_t @-> tensor @-> tensor @-> returning void)

  (** [get_device backend] returns the device associated with the backend.
      - [backend] The backend.
      - returns The backend device. *)
  let get_device = foreign (ns "get_device") (backend_t @-> returning dev_t)

  (** Events *)

  (** [event_new device] creates a new backend event associated with a device.
      - [device] The backend device.
      - returns The new event. *)
  let event_new = foreign (ns "event_new") (dev_t @-> returning event_t)

  (** [event_free event] frees the backend event.
      - [event] The event to free. *)
  let event_free = foreign (ns "event_free") (event_t @-> returning void)

  (** [event_record event backend] records the event in the backend's stream.
      - [event] The event to record.
      - [backend] The backend stream. *)
  let event_record = foreign (ns "event_record") (event_t @-> backend_t @-> returning void)

  (** [event_synchronize event] waits until the event has been signaled.
      - [event] The event to wait for. *)
  let event_synchronize = foreign (ns "event_synchronize") (event_t @-> returning void)

  (** [event_wait backend event] makes the backend stream wait for the event to be signaled.
      - [backend] The backend stream to wait.
      - [event] The event to wait for. *)
  let event_wait = foreign (ns "event_wait") (backend_t @-> event_t @-> returning void)

  (** Backend Device *)

  (** [dev_name device] returns the name of the backend device.
      - [device] The backend device.
      - returns The name. *)
  let dev_name = foreign (ns "dev_name") (dev_t @-> returning string)

  (** [dev_description device] returns a description of the backend device.
      - [device] The backend device.
      - returns The description. *)
  let dev_description = foreign (ns "dev_description") (dev_t @-> returning string)

  (** [dev_memory device free total] gets the free and total memory for the device.
      - [device] The backend device.
      - [free] Pointer to store the free memory in bytes.
      - [total] Pointer to store the total memory in bytes. *)
  let dev_memory = foreign (ns "dev_memory") (dev_t @-> ptr size_t @-> ptr size_t @-> returning void)

  (** [dev_type device] returns the type of the backend device (CPU, GPU, ACCEL).
      - [device] The backend device.
      - returns The device type enum. *)
  let dev_type = foreign (ns "dev_type") (dev_t @-> returning dev_type)

  (** [dev_get_props device props] gets the properties of the backend device.
      - [device] The backend device.
      - [props] Pointer to a `ggml_backend_dev_props` struct to fill. *)
  let dev_get_props = foreign (ns "dev_get_props") (dev_t @-> ptr DevProps.t @-> returning void)

  (** [dev_backend_reg device] returns the backend registry associated with the device.
      - [device] The backend device.
      - returns The backend registry. *)
  let dev_backend_reg = foreign (ns "dev_backend_reg") (dev_t @-> returning reg_t)

  (** [dev_init device params] initializes a backend instance from the device.
      - [device] The backend device.
      - [params] Optional initialization parameters string (can be NULL).
      - returns The initialized backend instance. *)
  let dev_init = foreign (ns "dev_init") (dev_t @-> string_opt @-> returning backend_t)
  (* string_opt for const char * params which can be NULL *)

  (** [dev_buffer_type device] returns the default buffer type for the device.
      - [device] The backend device.
      - returns The default buffer type. *)
  let dev_buffer_type = foreign (ns "dev_buffer_type") (dev_t @-> returning buffer_type_t)

  (** [dev_host_buffer_type device] returns the host buffer type for the device (if supported).
      - [device] The backend device.
      - returns The host buffer type, or NULL if not supported. *)
  let dev_host_buffer_type = foreign (ns "dev_host_buffer_type") (dev_t @-> returning buffer_type_t)

  (** [dev_buffer_from_host_ptr device ptr size max_tensor_size] creates a device buffer from a host pointer (if
      supported).
      - [device] The backend device.
      - [ptr] Pointer to the host memory.
      - [size] Size of the host memory region.
      - [max_tensor_size] Maximum size of a single tensor that can be allocated in this buffer.
      - returns The device buffer, or NULL if not supported or failed. *)
  let dev_buffer_from_host_ptr =
    foreign (ns "dev_buffer_from_host_ptr") (dev_t @-> ptr void @-> size_t @-> size_t @-> returning buffer_t)

  (** [dev_supports_op device op] checks if the device supports the operation of the given tensor.
      - [device] The backend device.
      - [op] The tensor representing the operation.
      - returns True if supported, false otherwise. *)
  let dev_supports_op = foreign (ns "dev_supports_op") (dev_t @-> const_tensor @-> returning bool)

  (** [dev_supports_buft device buft] checks if the device supports the given buffer type.
      - [device] The backend device.
      - [buft] The buffer type.
      - returns True if supported, false otherwise. *)
  let dev_supports_buft = foreign (ns "dev_supports_buft") (dev_t @-> buffer_type_t @-> returning bool)

  (** [dev_offload_op device op] checks if the operation should be offloaded to this device.
      - [device] The backend device.
      - [op] The tensor representing the operation.
      - returns True if it should be offloaded, false otherwise. *)
  let dev_offload_op = foreign (ns "dev_offload_op") (dev_t @-> const_tensor @-> returning bool)

  (** Backend Registry (reg) *)

  (** [reg_name reg] returns the name of the backend registry.
      - [reg] The backend registry.
      - returns The name. *)
  let reg_name = foreign (ns "reg_name") (reg_t @-> returning string)

  (** [reg_dev_count reg] returns the number of devices registered in this registry.
      - [reg] The backend registry.
      - returns The number of devices. *)
  let reg_dev_count = foreign (ns "reg_dev_count") (reg_t @-> returning size_t)

  (** [reg_dev_get reg index] gets a device from the registry by index.
      - [reg] The backend registry.
      - [index] The index of the device.
      - returns The backend device. *)
  let reg_dev_get = foreign (ns "reg_dev_get") (reg_t @-> size_t @-> returning dev_t)

  (** [reg_get_proc_address reg name] gets the address of a function exported by the backend registry.
      - [reg] The backend registry.
      - [name] The name of the function.
      - returns A pointer to the function, or NULL if not found. *)
  let reg_get_proc_address = foreign (ns "reg_get_proc_address") (reg_t @-> string @-> returning (ptr void))

  (** Backend Registry (Global) *)

  (** [device_register device] registers a backend device globally.
      - [device] The backend device to register. *)
  let device_register = foreign (ns "device_register") (dev_t @-> returning void)

  (** [reg_count ()] returns the total number of registered backend registries.
      - returns The number of registries. *)
  let reg_count = foreign (ns "reg_count") (void @-> returning size_t)

  (** [reg_get index] gets a backend registry by its global index.
      - [index] The index of the registry.
      - returns The backend registry. *)
  let reg_get = foreign (ns "reg_get") (size_t @-> returning reg_t)

  (** [reg_by_name name] finds a backend registry by name.
      - [name] The name of the registry.
      - returns The backend registry, or NULL if not found. *)
  let reg_by_name = foreign (ns "reg_by_name") (string @-> returning reg_t)

  (** [dev_count ()] returns the total number of registered backend devices globally.
      - returns The number of devices. *)
  let dev_count = foreign (ns "dev_count") (void @-> returning size_t)

  (** [dev_get index] gets a backend device by its global index.
      - [index] The index of the device.
      - returns The backend device. *)
  let dev_get = foreign (ns "dev_get") (size_t @-> returning dev_t)

  (** [dev_by_name name] finds a backend device by name globally.
      - [name] The name of the device.
      - returns The backend device, or NULL if not found. *)
  let dev_by_name = foreign (ns "dev_by_name") (string @-> returning dev_t)

  (** [dev_by_type type] finds the first backend device of the specified type globally.
      - [type] The device type.
      - returns The backend device, or NULL if not found. *)
  let dev_by_type = foreign (ns "dev_by_type") (Backend.dev_type @-> returning dev_t)

  (** [init_by_name name params] initializes a backend by device name. Shortcut for `dev_init(dev_by_name(name),
      params)`.
      - [name] The name of the device.
      - [params] Optional initialization parameters string.
      - returns The initialized backend instance, or NULL on failure. *)
  let init_by_name = foreign (ns "init_by_name") (string @-> string_opt @-> returning backend_t)

  (** [init_by_type type params] initializes a backend by device type. Shortcut for `dev_init(dev_by_type(type),
      params)`.
      - [type] The device type.
      - [params] Optional initialization parameters string.
      - returns The initialized backend instance, or NULL on failure. *)
  let init_by_type = foreign (ns "init_by_type") (Backend.dev_type @-> string_opt @-> returning backend_t)

  (** [init_best ()] initializes the best available backend (GPU first, then CPU).
      - returns The initialized backend instance. *)
  let init_best = foreign (ns "init_best") (void @-> returning backend_t)

  (** [load path] loads a backend from a dynamic library and registers it.
      - [path] Path to the dynamic library.
      - returns The loaded backend registry, or NULL on failure. *)
  let load = foreign (ns "load") (string @-> returning reg_t)

  (** [unload reg] unloads a dynamically loaded backend registry.
      - [reg] The backend registry to unload. *)
  let unload = foreign (ns "unload") (reg_t @-> returning void)

  (** [load_all ()] loads all known backend dynamic libraries from standard locations. *)
  let load_all = foreign (ns "load_all") (void @-> returning void)

  (** [load_all_from_path dir_path] loads all backend dynamic libraries from a specific directory.
      - [dir_path] Path to the directory containing backend libraries. *)
  let load_all_from_path = foreign (ns "load_all_from_path") (string @-> returning void)

  (** Backend Scheduler *)

  (** [sched_struct] Structure for the backend scheduler. *)
  let sched_struct : [ `Sched ] structure typ = structure (ns "sched")

  let sched_t = ptr sched_struct

  (** [sched_new backends bufts n_backends graph_size parallel] creates a new backend scheduler.
      - [backends] Array of backend instances.
      - [bufts] Array of corresponding buffer types (optional, can be NULL).
      - [n_backends] Number of backends in the arrays.
      - [graph_size] Estimated maximum graph size (nodes).
      - [parallel] Whether to allow parallel execution across backends.
      - [op_offload] Whether to enable operation offloading.
      - returns The new backend scheduler. *)
  let sched_new =
    foreign (ns "sched_new")
      (ptr backend_t @-> ptr buffer_type_t @-> int @-> size_t @-> bool @-> bool @-> returning sched_t)

  (** [sched_free sched] frees the backend scheduler.
      - [sched] The scheduler to free. *)
  let sched_free = foreign (ns "sched_free") (sched_t @-> returning void)

  (** [sched_reserve sched measure_graph] reserves backend buffers based on a measurement graph.
      - [sched] The scheduler.
      - [measure_graph] A graph representing the maximum expected memory usage.
      - returns True on success, false on failure. *)
  let sched_reserve = foreign (ns "sched_reserve") (sched_t @-> cgraph @-> returning bool)

  (** [sched_get_n_backends sched] returns the number of backends managed by the scheduler.
      - [sched] The scheduler.
      - returns The number of backends. *)
  let sched_get_n_backends = foreign (ns "sched_get_n_backends") (sched_t @-> returning int)

  (** [sched_get_backend sched i] gets a backend managed by the scheduler by index.
      - [sched] The scheduler.
      - [i] The index of the backend.
      - returns The backend instance. *)
  let sched_get_backend = foreign (ns "sched_get_backend") (sched_t @-> int @-> returning backend_t)

  (** [sched_get_n_splits sched] returns the number of graph splits from the last computation.
      - [sched] The scheduler.
      - returns The number of splits. *)
  let sched_get_n_splits = foreign (ns "sched_get_n_splits") (sched_t @-> returning int)

  (** [sched_get_n_copies sched] returns the number of tensor copies performed during the last computation.
      - [sched] The scheduler.
      - returns The number of copies. *)
  let sched_get_n_copies = foreign (ns "sched_get_n_copies") (sched_t @-> returning int)

  (** [sched_get_buffer_size sched backend] returns the size of the compute buffer allocated for a specific backend.
      - [sched] The scheduler.
      - [backend] The backend.
      - returns The buffer size in bytes. *)
  let sched_get_buffer_size = foreign (ns "sched_get_buffer_size") (sched_t @-> backend_t @-> returning size_t)

  (** [sched_set_tensor_backend sched node backend] manually assigns a graph node to a specific backend.
      - [sched] The scheduler.
      - [node] The tensor node to assign.
      - [backend] The target backend. *)
  let sched_set_tensor_backend =
    foreign (ns "sched_set_tensor_backend") (sched_t @-> tensor @-> backend_t @-> returning void)

  (** [sched_get_tensor_backend sched node] gets the backend assigned to a graph node.
      - [sched] The scheduler.
      - [node] The tensor node.
      - returns The assigned backend. *)
  let sched_get_tensor_backend = foreign (ns "sched_get_tensor_backend") (sched_t @-> tensor @-> returning backend_t)

  (** [sched_alloc_graph sched graph] allocates the compute buffers for the graph on the scheduler.
      - [sched] The scheduler.
      - [graph] The computation graph.
      - returns True on success, false on failure. *)
  let sched_alloc_graph = foreign (ns "sched_alloc_graph") (sched_t @-> cgraph @-> returning bool)

  (** [sched_graph_compute sched graph] computes the graph using the scheduler. Allocates buffers if not already done.
      - [sched] The scheduler.
      - [graph] The computation graph.
      - returns Status code. *)
  let sched_graph_compute = foreign (ns "sched_graph_compute") (sched_t @-> cgraph @-> returning status)

  (** [sched_graph_compute_async sched graph] asynchronously computes the graph using the scheduler.
      - [sched] The scheduler.
      - [graph] The computation graph.
      - returns Status code. *)
  let sched_graph_compute_async = foreign (ns "sched_graph_compute_async") (sched_t @-> cgraph @-> returning status)

  (** [sched_synchronize sched] synchronizes all backends managed by the scheduler.
      - [sched] The scheduler. *)
  let sched_synchronize = foreign (ns "sched_synchronize") (sched_t @-> returning void)

  (** [sched_reset sched] resets the scheduler's allocation state, deallocating graph tensors.
      - [sched] The scheduler. *)
  let sched_reset = foreign (ns "sched_reset") (sched_t @-> returning void)

  (** [sched_set_eval_callback sched callback user_data] sets a callback for observing graph nodes during computation.
      - [sched] The scheduler.
      - [callback] The evaluation callback function.
      - [user_data] User data passed to the callback. *)
  let sched_set_eval_callback =
    foreign (ns "sched_set_eval_callback") (sched_t @-> sched_eval_callback @-> ptr void @-> returning void)

  (** Backend Utils *)

  (** [graph_copy backend graph] copies a graph structure and its tensors to a specified backend.
      - [backend] The target backend.
      - [graph] The source graph.
      - returns A structure containing the copied graph and associated resources. *)
  let graph_copy = foreign (ns "graph_copy") (backend_t @-> cgraph @-> returning GraphCopy.t)

  (** [graph_copy_free copy] frees the resources associated with a graph copy.
      - [copy] The graph copy structure returned by `graph_copy`. *)
  let graph_copy_free = foreign (ns "graph_copy_free") (GraphCopy.t @-> returning void)

  (** [compare_graph_backend backend1 backend2 graph callback user_data] compares the output of a graph computed on two
      different backends.
      - [backend1] The first backend.
      - [backend2] The second backend.
      - [graph] The computation graph.
      - [callback] A callback function to compare individual node outputs.
      - [user_data] User data passed to the callback.
      - returns True if the outputs match, false otherwise. *)
  let compare_graph_backend =
    foreign
      (ns "compare_graph_backend")
      (backend_t @-> backend_t @-> cgraph @-> eval_callback @-> ptr void @-> tensor @-> returning bool)

  (** [tensor_alloc buffer tensor addr] allocates memory for a tensor within a buffer at a specific address. (Internal
      use likely).
      - [buffer] The buffer.
      - [tensor] The tensor.
      - [addr] The specific address within the buffer.
      - returns Status code. *)
  let tensor_alloc = foreign (ns "tensor_alloc") (buffer_t @-> tensor @-> ptr void @-> returning status)

  (** [view_init tensor] initializes the strides and data pointer for a view tensor based on its source. (Internal use
      likely).
      - [tensor] The view tensor to initialize.
      - returns Status code. *)
  let view_init = foreign (ns "view_init") (tensor @-> returning status)

  (** [cpu_buffer_from_ptr ptr size] creates a CPU backend buffer from an existing host memory pointer.
      - [ptr] Pointer to the host memory.
      - [size] Size of the memory region.
      - returns The CPU backend buffer. *)
  let cpu_buffer_from_ptr = foreign (ns "cpu_buffer_from_ptr") (ptr void @-> size_t @-> returning buffer_t)

  (** [cpu_buffer_type ()] returns the CPU backend buffer type.
      - returns The CPU buffer type. *)
  let cpu_buffer_type = foreign (ns "cpu_buffer_type") (void @-> returning buffer_type_t)
end
