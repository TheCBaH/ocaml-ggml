open Ctypes
module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types.GGUF

  (** [init_empty ()] creates an empty GGUF context.
      @return A new GGUF context. *)
  let init_empty = foreign (ns "init_empty") (void @-> returning context_t)

  (** [init_from_file fname params] initializes a GGUF context from a file.
      @param fname Path to the GGUF file.
      @param params Initialization parameters.
      @return The loaded GGUF context. *)
  let init_from_file = foreign (ns "init_from_file") (string @-> InitParams.t @-> returning context_t)

  (** [free ctx] frees the memory associated with a GGUF context.
      @param ctx The GGUF context to free. *)
  let free = foreign (ns "free") (context_t @-> returning void)

  (** [type_name typ] returns the name of the GGUF type.
      @param typ The GGUF type enum value.
      @return The name of the type. *)
  let type_name = foreign (ns "type_name") (typ @-> returning string)

  (** [get_version ctx] returns the version of the GGUF file format used by the context.
      @param ctx The GGUF context.
      @return The GGUF version number. *)
  let get_version = foreign (ns "get_version") (context_t @-> returning uint32_t)

  (** [get_alignment ctx] returns the alignment used for tensor data in the GGUF context.
      @param ctx The GGUF context.
      @return The alignment value in bytes. *)
  let get_alignment = foreign (ns "get_alignment") (context_t @-> returning size_t)

  (** [get_data_offset ctx] returns the offset in bytes to the start of the tensor data blob.
      @param ctx The GGUF context.
      @return The data offset. *)
  let get_data_offset = foreign (ns "get_data_offset") (context_t @-> returning size_t)

  (** [get_n_kv ctx] returns the number of key-value pairs in the GGUF context.
      @param ctx The GGUF context.
      @return The number of key-value pairs. *)
  let get_n_kv = foreign (ns "get_n_kv") (context_t @-> returning int64_t)

  (** [find_key ctx key] finds the index of a key in the GGUF context.
      @param ctx The GGUF context.
      @param key The key string to find.
      @return The index of the key, or -1 if not found. *)
  let find_key = foreign (ns "find_key") (context_t @-> string @-> returning int64_t)

  (** [get_key ctx key_id] returns the key string at the specified index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The key string. *)
  let get_key = foreign (ns "get_key") (context_t @-> int64_t @-> returning string)

  (** [get_kv_type ctx key_id] returns the GGUF type of the value associated with the key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The GGUF type enum value. *)
  let get_kv_type = foreign (ns "get_kv_type") (context_t @-> int64_t @-> returning typ)

  (** [get_arr_type ctx key_id] returns the GGUF type of the elements in an array value.
      @param ctx The GGUF context.
      @param key_id The index of the key (must be of type GGUF_TYPE_ARRAY).
      @return The GGUF type enum value of the array elements. *)
  let get_arr_type = foreign (ns "get_arr_type") (context_t @-> int64_t @-> returning typ)

  (** [get_val_u8 ctx key_id] gets the uint8_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The uint8_t value. *)
  let get_val_u8 = foreign (ns "get_val_u8") (context_t @-> int64_t @-> returning uint8_t)

  (** [get_val_i8 ctx key_id] gets the int8_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The int8_t value. *)
  let get_val_i8 = foreign (ns "get_val_i8") (context_t @-> int64_t @-> returning int8_t)

  (** [get_val_u16 ctx key_id] gets the uint16_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The uint16_t value. *)
  let get_val_u16 = foreign (ns "get_val_u16") (context_t @-> int64_t @-> returning uint16_t)

  (** [get_val_i16 ctx key_id] gets the int16_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The int16_t value. *)
  let get_val_i16 = foreign (ns "get_val_i16") (context_t @-> int64_t @-> returning int16_t)

  (** [get_val_u32 ctx key_id] gets the uint32_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The uint32_t value. *)
  let get_val_u32 = foreign (ns "get_val_u32") (context_t @-> int64_t @-> returning uint32_t)

  (** [get_val_i32 ctx key_id] gets the int32_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The int32_t value. *)
  let get_val_i32 = foreign (ns "get_val_i32") (context_t @-> int64_t @-> returning int32_t)

  (** [get_val_f32 ctx key_id] gets the float value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The float value. *)
  let get_val_f32 = foreign (ns "get_val_f32") (context_t @-> int64_t @-> returning float)

  (** [get_val_u64 ctx key_id] gets the uint64_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The uint64_t value. *)
  let get_val_u64 = foreign (ns "get_val_u64") (context_t @-> int64_t @-> returning uint64_t)

  (** [get_val_i64 ctx key_id] gets the int64_t value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The int64_t value. *)
  let get_val_i64 = foreign (ns "get_val_i64") (context_t @-> int64_t @-> returning int64_t)

  (** [get_val_f64 ctx key_id] gets the double value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The double value. *)
  let get_val_f64 = foreign (ns "get_val_f64") (context_t @-> int64_t @-> returning double)

  (** [get_val_bool ctx key_id] gets the boolean value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The boolean value. *)
  let get_val_bool = foreign (ns "get_val_bool") (context_t @-> int64_t @-> returning bool)

  (** [get_val_str ctx key_id] gets the string value for the given key index.
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return The string value. *)
  let get_val_str = foreign (ns "get_val_str") (context_t @-> int64_t @-> returning string)

  (** [get_val_data ctx key_id] gets a raw pointer to the data for the given key index (non-array types).
      @param ctx The GGUF context.
      @param key_id The index of the key.
      @return A void pointer to the value data. *)
  let get_val_data = foreign (ns "get_val_data") (context_t @-> int64_t @-> returning (ptr (const void)))

  (** [get_arr_n ctx key_id] gets the number of elements in an array value.
      @param ctx The GGUF context.
      @param key_id The index of the key (must be of type GGUF_TYPE_ARRAY).
      @return The number of elements in the array. *)
  let get_arr_n = foreign (ns "get_arr_n") (context_t @-> int64_t @-> returning size_t)

  (** [get_arr_data ctx key_id] gets a raw pointer to the first element of an array value.
      @param ctx The GGUF context.
      @param key_id The index of the key (must be of type GGUF_TYPE_ARRAY).
      @return A void pointer to the array data. *)
  let get_arr_data = foreign (ns "get_arr_data") (context_t @-> int64_t @-> returning (ptr (const void)))

  (** [get_arr_str ctx key_id i] gets the i-th string from a string array value.
      @param ctx The GGUF context.
      @param key_id The index of the key (must be an array of strings).
      @param i The index of the string within the array.
      @return The string value. *)
  let get_arr_str = foreign (ns "get_arr_str") (context_t @-> int64_t @-> size_t @-> returning string)

  (** [get_n_tensors ctx] returns the number of tensors in the GGUF context.
      @param ctx The GGUF context.
      @return The number of tensors. *)
  let get_n_tensors = foreign (ns "get_n_tensors") (context_t @-> returning int64_t)

  (** [find_tensor ctx name] finds the index of a tensor by its name.
      @param ctx The GGUF context.
      @param name The name of the tensor.
      @return The index of the tensor, or -1 if not found. *)
  let find_tensor = foreign (ns "find_tensor") (context_t @-> string @-> returning int64_t)

  (** [get_tensor_offset ctx tensor_id] returns the offset in bytes of the tensor data within the data blob.
      @param ctx The GGUF context.
      @param tensor_id The index of the tensor.
      @return The offset in bytes. *)
  let get_tensor_offset = foreign (ns "get_tensor_offset") (context_t @-> int64_t @-> returning size_t)

  (** [get_tensor_name ctx tensor_id] returns the name of the tensor at the specified index.
      @param ctx The GGUF context.
      @param tensor_id The index of the tensor.
      @return The name of the tensor. *)
  let get_tensor_name = foreign (ns "get_tensor_name") (context_t @-> int64_t @-> returning string)

  (** [get_tensor_type ctx tensor_id] returns the ggml type of the tensor at the specified index.
      @param ctx The GGUF context.
      @param tensor_id The index of the tensor.
      @return The ggml type enum value. *)
  let get_tensor_type = foreign (ns "get_tensor_type") (context_t @-> int64_t @-> returning Types.typ)
  (* ggml_type *)

  (** [get_tensor_size ctx tensor_id] returns the size in bytes of the tensor data.
      @param ctx The GGUF context.
      @param tensor_id The index of the tensor.
      @return The size in bytes. *)
  let get_tensor_size = foreign (ns "get_tensor_size") (context_t @-> int64_t @-> returning size_t)

  (** [remove_key ctx key] removes a key-value pair from the context.
      @param ctx The GGUF context.
      @param key The key to remove.
      @return The index the key had before removal, or -1 if it didn't exist. *)
  let remove_key = foreign (ns "remove_key") (context_t @-> string @-> returning int64_t)

  (** [set_val_u8 ctx key val] sets a uint8_t value for a key (adds or overrides).
      @param ctx The GGUF context.
      @param key The key string.
      @param val The uint8_t value. *)
  let set_val_u8 = foreign (ns "set_val_u8") (context_t @-> string @-> uint8_t @-> returning void)

  (** [set_val_i8 ctx key val] sets an int8_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The int8_t value. *)
  let set_val_i8 = foreign (ns "set_val_i8") (context_t @-> string @-> int8_t @-> returning void)

  (** [set_val_u16 ctx key val] sets a uint16_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The uint16_t value. *)
  let set_val_u16 = foreign (ns "set_val_u16") (context_t @-> string @-> uint16_t @-> returning void)

  (** [set_val_i16 ctx key val] sets an int16_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The int16_t value. *)
  let set_val_i16 = foreign (ns "set_val_i16") (context_t @-> string @-> int16_t @-> returning void)

  (** [set_val_u32 ctx key val] sets a uint32_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The uint32_t value. *)
  let set_val_u32 = foreign (ns "set_val_u32") (context_t @-> string @-> uint32_t @-> returning void)

  (** [set_val_i32 ctx key val] sets an int32_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The int32_t value. *)
  let set_val_i32 = foreign (ns "set_val_i32") (context_t @-> string @-> int32_t @-> returning void)

  (** [set_val_f32 ctx key val] sets a float value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The float value. *)
  let set_val_f32 = foreign (ns "set_val_f32") (context_t @-> string @-> float @-> returning void)

  (** [set_val_u64 ctx key val] sets a uint64_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The uint64_t value. *)
  let set_val_u64 = foreign (ns "set_val_u64") (context_t @-> string @-> uint64_t @-> returning void)

  (** [set_val_i64 ctx key val] sets an int64_t value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The int64_t value. *)
  let set_val_i64 = foreign (ns "set_val_i64") (context_t @-> string @-> int64_t @-> returning void)

  (** [set_val_f64 ctx key val] sets a double value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The double value. *)
  let set_val_f64 = foreign (ns "set_val_f64") (context_t @-> string @-> double @-> returning void)

  (** [set_val_bool ctx key val] sets a boolean value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The boolean value. *)
  let set_val_bool = foreign (ns "set_val_bool") (context_t @-> string @-> bool @-> returning void)

  (** [set_val_str ctx key val] sets a string value for a key.
      @param ctx The GGUF context.
      @param key The key string.
      @param val The string value. *)
  let set_val_str = foreign (ns "set_val_str") (context_t @-> string @-> string @-> returning void)

  (** [set_arr_data ctx key typ data n] sets an array value with raw data.
      @param ctx The GGUF context.
      @param key The key string.
      @param typ The GGUF type of the array elements.
      @param data A void pointer to the array data.
      @param n The number of elements in the array. *)
  let set_arr_data =
    foreign (ns "set_arr_data") (context_t @-> string @-> typ @-> ptr void @-> size_t @-> returning void)

  (** [set_arr_str ctx key data n] sets an array of strings value.
      @param ctx The GGUF context.
      @param key The key string.
      @param data A pointer to an array of C strings.
      @param n The number of strings in the array. *)
  let set_arr_str = foreign (ns "set_arr_str") (context_t @-> string @-> ptr string @-> size_t @-> returning void)

  (** [set_kv ctx src] copies all key-value pairs from context `src` to `ctx`.
      @param ctx The destination GGUF context.
      @param src The source GGUF context. *)
  let set_kv = foreign (ns "set_kv") (context_t @-> context_t @-> returning void)

  (** [add_tensor ctx tensor] adds a ggml tensor's information to the GGUF context.
      @param ctx The GGUF context.
      @param tensor The ggml tensor to add. *)
  let add_tensor = foreign (ns "add_tensor") (context_t @-> Types.tensor @-> returning void)
  (* ggml_tensor *)

  (** [set_tensor_type ctx name typ] changes the ggml type associated with a tensor name in the GGUF context.
      @param ctx The GGUF context.
      @param name The name of the tensor.
      @param typ The new ggml type. *)
  let set_tensor_type = foreign (ns "set_tensor_type") (context_t @-> string @-> Types.typ @-> returning void)
  (* ggml_type *)

  (** [set_tensor_data ctx name data] sets the raw data pointer associated with a tensor name (for writing).
      @param ctx The GGUF context.
      @param name The name of the tensor.
      @param data A void pointer to the tensor data. *)
  let set_tensor_data = foreign (ns "set_tensor_data") (context_t @-> string @-> ptr void @-> returning void)

  (** [write_to_file ctx fname only_meta] writes the GGUF context to a file.
      @param ctx The GGUF context.
      @param fname The output filename.
      @param only_meta If true, only write metadata (header, KV, tensor info); if false, write metadata and tensor data.
      @return True on success, false on failure. *)
  let write_to_file = foreign (ns "write_to_file") (context_t @-> string @-> bool @-> returning bool)

  (** [get_meta_size ctx] calculates the size in bytes of the metadata section (including padding).
      @param ctx The GGUF context.
      @return The size of the metadata in bytes. *)
  let get_meta_size = foreign (ns "get_meta_size") (context_t @-> returning size_t)

  (** [get_meta_data ctx data] writes the metadata section to the provided buffer.
      @param ctx The GGUF context.
      @param data A void pointer to a buffer large enough to hold the metadata (size obtained from `get_meta_size`). *)
  let get_meta_data = foreign (ns "get_meta_data") (context_t @-> ptr void @-> returning void)
end
