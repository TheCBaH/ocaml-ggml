open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated
  open Types_generated.GGUF

  (** [init_empty ()] creates an empty GGUF context. Returns a new GGUF context. *)
  let init_empty = foreign (ns "init_empty") (void @-> returning context_t)

  (** [init_from_file fname params] initializes a GGUF context from a file.
      - [fname] Path to the GGUF file.
      - [params] Initialization parameters.
      - returns The loaded GGUF context. *)
  let init_from_file = foreign (ns "init_from_file") (string @-> InitParams.t @-> returning context_t)

  (** [free ctx] frees the memory associated with a GGUF context.
      - [ctx] The GGUF context to free. *)
  let free = foreign (ns "free") (context_t @-> returning void)

  (** [type_name typ] returns the name of the GGUF type.
      - [typ] The GGUF type enum value.
      - returns The name of the type. *)
  let type_name = foreign (ns "type_name") (typ @-> returning string)

  (** [get_version ctx] returns the version of the GGUF file format used by the context.
      - [ctx] The GGUF context.
      - returns The GGUF version number. *)
  let get_version = foreign (ns "get_version") (context_t @-> returning uint32_t)

  (** [get_alignment ctx] returns the alignment used for tensor data in the GGUF context.
      - [ctx] The GGUF context.
      - returns The alignment value in bytes. *)
  let get_alignment = foreign (ns "get_alignment") (context_t @-> returning size_t)

  (** [get_data_offset ctx] returns the offset in bytes to the start of the tensor data blob.
      - [ctx] The GGUF context.
      - returns The data offset. *)
  let get_data_offset = foreign (ns "get_data_offset") (context_t @-> returning size_t)

  (** [get_n_kv ctx] returns the number of key-value pairs in the GGUF context.
      - [ctx] The GGUF context.
      - returns The number of key-value pairs. *)
  let get_n_kv = foreign (ns "get_n_kv") (context_t @-> returning int64_t)

  (** [find_key ctx key] finds the index of a key in the GGUF context. Returns the index of the key, or -1 if not found.
      - [ctx] The GGUF context.
      - [key] The key string to find.
      - returns The index of the key, or -1 if not found. *)
  let find_key = foreign (ns "find_key") (context_t @-> string @-> returning int64_t)

  (** [get_key ctx key_id] returns the key string at the specified index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The key string. *)
  let get_key = foreign (ns "get_key") (context_t @-> int64_t @-> returning string)

  (** [get_kv_type ctx key_id] returns the GGUF type of the value associated with the key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The GGUF type enum value. *)
  let get_kv_type = foreign (ns "get_kv_type") (context_t @-> int64_t @-> returning typ)

  (** [get_arr_type ctx key_id] returns the GGUF type of the elements in an array value.
      - [ctx] The GGUF context.
      - [key_id] The index of the key (must be of type GGUF_TYPE_ARRAY).
      - returns The GGUF type enum value of the array elements. *)
  let get_arr_type = foreign (ns "get_arr_type") (context_t @-> int64_t @-> returning typ)

  (** [get_val_u8 ctx key_id] gets the uint8_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The uint8_t value. *)
  let get_val_u8 = foreign (ns "get_val_u8") (context_t @-> int64_t @-> returning uint8_t)

  (** [get_val_i8 ctx key_id] gets the int8_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The int8_t value. *)
  let get_val_i8 = foreign (ns "get_val_i8") (context_t @-> int64_t @-> returning int8_t)

  (** [get_val_u16 ctx key_id] gets the uint16_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The uint16_t value. *)
  let get_val_u16 = foreign (ns "get_val_u16") (context_t @-> int64_t @-> returning uint16_t)

  (** [get_val_i16 ctx key_id] gets the int16_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The int16_t value. *)
  let get_val_i16 = foreign (ns "get_val_i16") (context_t @-> int64_t @-> returning int16_t)

  (** [get_val_u32 ctx key_id] gets the uint32_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The uint32_t value. *)
  let get_val_u32 = foreign (ns "get_val_u32") (context_t @-> int64_t @-> returning uint32_t)

  (** [get_val_i32 ctx key_id] gets the int32_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The int32_t value. *)
  let get_val_i32 = foreign (ns "get_val_i32") (context_t @-> int64_t @-> returning int32_t)

  (** [get_val_f32 ctx key_id] gets the float value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The float value. *)
  let get_val_f32 = foreign (ns "get_val_f32") (context_t @-> int64_t @-> returning float)

  (** [get_val_u64 ctx key_id] gets the uint64_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The uint64_t value. *)
  let get_val_u64 = foreign (ns "get_val_u64") (context_t @-> int64_t @-> returning uint64_t)

  (** [get_val_i64 ctx key_id] gets the int64_t value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The int64_t value. *)
  let get_val_i64 = foreign (ns "get_val_i64") (context_t @-> int64_t @-> returning int64_t)

  (** [get_val_f64 ctx key_id] gets the double value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The double value. *)
  let get_val_f64 = foreign (ns "get_val_f64") (context_t @-> int64_t @-> returning double)

  (** [get_val_bool ctx key_id] gets the boolean value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The boolean value. *)
  let get_val_bool = foreign (ns "get_val_bool") (context_t @-> int64_t @-> returning bool)

  (** [get_val_str ctx key_id] gets the string value for the given key index.
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns The string value. *)
  let get_val_str = foreign (ns "get_val_str") (context_t @-> int64_t @-> returning string)

  (** [get_val_data ctx key_id] gets a raw pointer to the data for the given key index (non-array types).
      - [ctx] The GGUF context.
      - [key_id] The index of the key.
      - returns A void pointer to the value data. *)
  let get_val_data = foreign (ns "get_val_data") (context_t @-> int64_t @-> returning (ptr (const void)))

  (** [get_arr_n ctx key_id] gets the number of elements in an array value.
      - [ctx] The GGUF context.
      - [key_id] The index of the key (must be of type GGUF_TYPE_ARRAY).
      - returns The number of elements in the array. *)
  let get_arr_n = foreign (ns "get_arr_n") (context_t @-> int64_t @-> returning size_t)

  (** [get_arr_data ctx key_id] gets a raw pointer to the first element of an array value.
      - [ctx] The GGUF context.
      - [key_id] The index of the key (must be of type GGUF_TYPE_ARRAY).
      - returns A void pointer to the array data. *)
  let get_arr_data = foreign (ns "get_arr_data") (context_t @-> int64_t @-> returning (ptr (const void)))

  (** [get_arr_str ctx key_id i] gets the i-th string from a string array value.
      - [ctx] The GGUF context.
      - [key_id] The index of the key (must be an array of strings).
      - [i] The index of the string within the array.
      - returns The string value. *)
  let get_arr_str = foreign (ns "get_arr_str") (context_t @-> int64_t @-> size_t @-> returning string)

  (** [get_n_tensors ctx] returns the number of tensors in the GGUF context.
      - [ctx] The GGUF context.
      - returns The number of tensors. *)
  let get_n_tensors = foreign (ns "get_n_tensors") (context_t @-> returning int64_t)

  (** [find_tensor ctx name] finds the index of a tensor by its name. Returns the index of the tensor, or -1 if not
      found.
      - [ctx] The GGUF context.
      - [name] The name of the tensor.
      - returns The index of the tensor, or -1 if not found. *)
  let find_tensor = foreign (ns "find_tensor") (context_t @-> string @-> returning int64_t)

  (** [get_tensor_offset ctx tensor_id] returns the offset in bytes of the tensor data within the data blob.
      - [ctx] The GGUF context.
      - [tensor_id] The index of the tensor.
      - returns The offset in bytes. *)
  let get_tensor_offset = foreign (ns "get_tensor_offset") (context_t @-> int64_t @-> returning size_t)

  (** [get_tensor_name ctx tensor_id] returns the name of the tensor at the specified index.
      - [ctx] The GGUF context.
      - [tensor_id] The index of the tensor.
      - returns The name of the tensor. *)
  let get_tensor_name = foreign (ns "get_tensor_name") (context_t @-> int64_t @-> returning string)

  (** [get_tensor_type ctx tensor_id] returns the ggml type of the tensor at the specified index.
      - [ctx] The GGUF context.
      - [tensor_id] The index of the tensor.
      - returns The ggml type enum value. *)
  let get_tensor_type = foreign (ns "get_tensor_type") (context_t @-> int64_t @-> returning typ)
  (* ggml_type *)

  (** [get_tensor_size ctx tensor_id] returns the size in bytes of the tensor data.
      - [ctx] The GGUF context.
      - [tensor_id] The index of the tensor.
      - returns The size in bytes. *)
  let get_tensor_size = foreign (ns "get_tensor_size") (context_t @-> int64_t @-> returning size_t)

  (** [remove_key ctx key] removes a key-value pair from the context.
      - [ctx] The GGUF context.
      - [key] The key to remove.
      - returns The index the key had before removal, or -1 if it didn't exist. *)
  let remove_key = foreign (ns "remove_key") (context_t @-> string @-> returning int64_t)

  (** [set_val_u8 ctx key val] sets a uint8_t value for a key (adds or overrides).
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The uint8_t value. *)
  let set_val_u8 = foreign (ns "set_val_u8") (context_t @-> string @-> uint8_t @-> returning void)

  (** [set_val_i8 ctx key val] sets an int8_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The int8_t value. *)
  let set_val_i8 = foreign (ns "set_val_i8") (context_t @-> string @-> int8_t @-> returning void)

  (** [set_val_u16 ctx key val] sets a uint16_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The uint16_t value. *)
  let set_val_u16 = foreign (ns "set_val_u16") (context_t @-> string @-> uint16_t @-> returning void)

  (** [set_val_i16 ctx key val] sets an int16_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The int16_t value. *)
  let set_val_i16 = foreign (ns "set_val_i16") (context_t @-> string @-> int16_t @-> returning void)

  (** [set_val_u32 ctx key val] sets a uint32_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The uint32_t value. *)
  let set_val_u32 = foreign (ns "set_val_u32") (context_t @-> string @-> uint32_t @-> returning void)

  (** [set_val_i32 ctx key val] sets an int32_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The int32_t value. *)
  let set_val_i32 = foreign (ns "set_val_i32") (context_t @-> string @-> int32_t @-> returning void)

  (** [set_val_f32 ctx key val] sets a float value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The float value. *)
  let set_val_f32 = foreign (ns "set_val_f32") (context_t @-> string @-> float @-> returning void)

  (** [set_val_u64 ctx key val] sets a uint64_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The uint64_t value. *)
  let set_val_u64 = foreign (ns "set_val_u64") (context_t @-> string @-> uint64_t @-> returning void)

  (** [set_val_i64 ctx key val] sets an int64_t value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The int64_t value. *)
  let set_val_i64 = foreign (ns "set_val_i64") (context_t @-> string @-> int64_t @-> returning void)

  (** [set_val_f64 ctx key val] sets a double value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The double value. *)
  let set_val_f64 = foreign (ns "set_val_f64") (context_t @-> string @-> double @-> returning void)

  (** [set_val_bool ctx key val] sets a boolean value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The boolean value. *)
  let set_val_bool = foreign (ns "set_val_bool") (context_t @-> string @-> bool @-> returning void)

  (** [set_val_str ctx key val] sets a string value for a key.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [val] The string value. *)
  let set_val_str = foreign (ns "set_val_str") (context_t @-> string @-> string @-> returning void)

  (** [set_arr_data ctx key typ data n] sets an array value with raw data.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [typ] The GGUF type of the array elements.
      - [data] A void pointer to the array data.
      - [n] The number of elements in the array. *)
  let set_arr_data =
    foreign (ns "set_arr_data") (context_t @-> string @-> typ @-> ptr void @-> size_t @-> returning void)

  (** [set_arr_str ctx key data n] sets an array of strings value.
      - [ctx] The GGUF context.
      - [key] The key string.
      - [data] A pointer to an array of C strings.
      - [n] The number of strings in the array. *)
  let set_arr_str = foreign (ns "set_arr_str") (context_t @-> string @-> ptr string @-> size_t @-> returning void)

  (** [set_kv ctx src] copies all key-value pairs from context `src` to `ctx`.
      - [ctx] The destination GGUF context.
      - [src] The source GGUF context. *)
  let set_kv = foreign (ns "set_kv") (context_t @-> context_t @-> returning void)

  (** [add_tensor ctx tensor] adds a ggml tensor's information to the GGUF context.
      - [ctx] The GGUF context.
      - [tensor] The ggml tensor to add. *)
  let add_tensor = foreign (ns "add_tensor") (context_t @-> tensor @-> returning void)
  (* ggml_tensor *)

  (** [set_tensor_type ctx name typ] changes the ggml type associated with a tensor name in the GGUF context.
      - [ctx] The GGUF context.
      - [name] The name of the tensor.
      - [typ] The new ggml type. *)
  let set_tensor_type = foreign (ns "set_tensor_type") (context_t @-> string @-> typ @-> returning void)
  (* ggml_type *)

  (** [set_tensor_data ctx name data] sets the raw data pointer associated with a tensor name (for writing).
      - [ctx] The GGUF context.
      - [name] The name of the tensor.
      - [data] A void pointer to the tensor data. *)
  let set_tensor_data = foreign (ns "set_tensor_data") (context_t @-> string @-> ptr void @-> returning void)

  (** [write_to_file ctx fname only_meta] writes the GGUF context to a file.
      - [ctx] The GGUF context.
      - [fname] The output filename.
      - [only_meta] If true, only write metadata (header, KV, tensor info); if false, write metadata and tensor data.
      - returns True on success, false on failure. *)
  let write_to_file = foreign (ns "write_to_file") (context_t @-> string @-> bool @-> returning bool)

  (** [get_meta_size ctx] calculates the size in bytes of the metadata section (including padding).
      - [ctx] The GGUF context.
      - returns The size of the metadata in bytes. *)
  let get_meta_size = foreign (ns "get_meta_size") (context_t @-> returning size_t)

  (** [get_meta_data ctx data] writes the metadata section to the provided buffer.
      - [ctx] The GGUF context.
      - [data] A void pointer to a buffer large enough to hold the metadata (size obtained from `get_meta_size`). *)
  let get_meta_data = foreign (ns "get_meta_data") (context_t @-> ptr void @-> returning void)
end
