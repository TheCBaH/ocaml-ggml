open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated

  (* Context *)

  (** [init params] initializes a ggml context.
      - [params] Initialization parameters.
      - returns A new ggml context. *)
  let init = foreign (ns "init") (InitParams.t @-> returning context)

  (** [free ctx] frees the memory associated with a ggml context.
      - [ctx] The context to free. *)
  let free = foreign (ns "free") (context @-> returning void)

  (** [used_mem ctx] returns the amount of memory used by the context in bytes.
      - [ctx] The context.
      - returns Memory used in bytes. *)
  let used_mem = foreign (ns "used_mem") (context @-> returning size_t)

  (** [reset ctx] resets the context, clearing its internal state but keeping allocated memory.
      - [ctx] The context to reset. *)
  let reset = foreign (ns "reset") (context @-> returning void)

  (** [guid_matches guid_a guid_b] checks if two GUIDs are equal.
      - [guid_a] First GUID.
      - [guid_b] Second GUID.
      - returns True if the GUIDs match, false otherwise. *)
  let guid_matches = foreign (ns "guid_matches") (guid_t @-> guid_t @-> returning bool)

  (* Misc *)

  (* Time Functions *)

  (** [time_init ()] initializes the internal timer. Call once at program start. *)
  let time_init = foreign (ns "time_init") (void @-> returning void)

  (** [time_ms ()] returns the current time in milliseconds.
      - returns Time in milliseconds. *)
  let time_ms = foreign (ns "time_ms") (void @-> returning int64_t)

  (** [time_us ()] returns the current time in microseconds.
      - returns Time in microseconds. *)
  let time_us = foreign (ns "time_us") (void @-> returning int64_t)

  (** [cycles ()] returns the current CPU cycle count.
      - returns CPU cycle count. *)
  let cycles = foreign (ns "cycles") (void @-> returning int64_t)

  (** [cycles_per_ms ()] returns the number of CPU cycles per millisecond.
      - returns CPU cycles per millisecond. *)
  let cycles_per_ms = foreign (ns "cycles_per_ms") (void @-> returning int64_t)

  (* File Handling *)

  (** [fopen fname mode] opens a file, accepting UTF-8 paths even on Windows.
      - [fname] The filename (UTF-8).
      - [mode] The file opening mode (e.g., "rb", "wb").
      - returns A file pointer (represented as `ptr void`), or NULL on failure. *)
  let fopen = foreign (ns "fopen") (string @-> string @-> returning (ptr void))

  (* Printing *)

  (** [print_object obj] prints information about a ggml object to stderr.
      - [obj] The object to print. *)
  let print_object = foreign (ns "print_object") (object' @-> returning void)

  (** [print_objects ctx] prints information about all objects in a context to stderr.
      - [ctx] The context. *)
  let print_objects = foreign (ns "print_objects") (context @-> returning void)

  (* Types / Ops Info *)

  (** [type_name typ] returns the name of the ggml type.
      - [typ] The ggml type.
      - returns The name of the type. *)
  let type_name = foreign (ns "type_name") (typ @-> returning string)

  (** [op_name op] returns the name of the ggml operation.
      - [op] The ggml operation.
      - returns The name of the operation. *)
  let op_name = foreign (ns "op_name") (op @-> returning string)

  (** [op_symbol op] returns the symbolic representation of the ggml operation (e.g., "+", "*").
      - [op] The ggml operation.
      - returns The symbol of the operation. *)
  let op_symbol = foreign (ns "op_symbol") (op @-> returning string)

  (** [unary_op_name op] returns the name of the ggml unary operation.
      - [op] The ggml unary operation.
      - returns The name of the unary operation. *)
  let unary_op_name = foreign (ns "unary_op_name") (unary_op @-> returning string)

  (** [glu_op_name op] returns the name of the ggml GLU operation.
      - [op] The ggml GLU operation.
      - returns The name of the GLU operation. *)
  let glu_op_name = foreign (ns "glu_op_name") (glu_op @-> returning string)

  (** [op_desc tensor] returns a description of the operation that produced the tensor.
      - [tensor] The tensor.
      - returns A description string (unary op name or op name). *)
  let op_desc = foreign (ns "op_desc") (tensor @-> returning string)

  (** [blck_size typ] returns the block size for a given ggml type.
      - [typ] The ggml type.
      - returns The block size (number of elements in a block). *)
  let blck_size = foreign (ns "blck_size") (typ @-> returning int64_t)

  (** [type_size typ] returns the size in bytes for all elements in a block of the given type.
      - [typ] The ggml type.
      - returns The size in bytes of a block. *)
  let type_size = foreign (ns "type_size") (typ @-> returning size_t)

  (** [row_size typ ne] returns the size in bytes for a row containing `ne` elements of the given type.
      - [typ] The ggml type.
      - [ne] The number of elements in the row.
      - returns The size in bytes of the row. *)
  let row_size = foreign (ns "row_size") (typ @-> int64_t @-> returning size_t)

  (** [is_quantized typ] checks if the ggml type is a quantized type.
      - [typ] The ggml type.
      - returns True if the type is quantized, false otherwise. *)
  let is_quantized = foreign (ns "is_quantized") (typ @-> returning bool)

  (** [ftype_to_ggml_type ftype] converts a file type enum to a ggml type enum.
      - [ftype] The file type enum value.
      - returns The corresponding ggml type enum value. *)
  let ftype_to_ggml_type = foreign (ns "ftype_to_ggml_type") (ftype @-> returning typ)

  (* Tensor Info *)

  (** [element_size tensor] returns the size in bytes of a single element in the tensor.
      - [tensor] The tensor.
      - returns The element size in bytes. *)
  let element_size = foreign (ns "element_size") (tensor @-> returning size_t)

  (** [nelements tensor] returns the total number of elements in the tensor.
      - [tensor] The tensor.
      - returns The number of elements. *)
  let nelements = foreign (ns "nelements") (tensor @-> returning int64_t)

  (** [nbytes tensor] returns the total size in bytes of the tensor's data.
      - [tensor] The tensor.
      - returns The size in bytes. *)
  let nbytes = foreign (ns "nbytes") (tensor @-> returning size_t)

  (** [nbytes_pad tensor] returns the padded size in bytes of the tensor's data (aligned to GGML_MEM_ALIGN).
      - [tensor] The tensor.
      - returns The padded size in bytes. *)
  let nbytes_pad = foreign (ns "nbytes_pad") (tensor @-> returning size_t)

  (** [nrows tensor] returns the number of rows in the tensor (product of dimensions >= 1).
      - [tensor] The tensor.
      - returns The number of rows. *)
  let nrows = foreign (ns "nrows") (tensor @-> returning int64_t)

  (** [is_transposed tensor] checks if the tensor is transposed (swapped strides for dims 0 and 1).
      - [tensor] The tensor.
      - returns True if transposed, false otherwise. *)
  let is_transposed = foreign (ns "is_transposed") (tensor @-> returning bool)

  (** [is_permuted tensor] checks if the tensor is permuted (strides differ from canonical order).
      - [tensor] The tensor.
      - returns True if permuted, false otherwise. *)
  let is_permuted = foreign (ns "is_permuted") (tensor @-> returning bool)

  (** [is_empty tensor] checks if the tensor has zero elements.
      - [tensor] The tensor.
      - returns True if empty, false otherwise. *)
  let is_empty = foreign (ns "is_empty") (tensor @-> returning bool)

  (** [is_scalar tensor] checks if the tensor is a scalar (all dimensions are 1).
      - [tensor] The tensor.
      - returns True if scalar, false otherwise. *)
  let is_scalar = foreign (ns "is_scalar") (tensor @-> returning bool)

  (** [is_vector tensor] checks if the tensor is a vector (one non-unity dimension).
      - [tensor] The tensor.
      - returns True if vector, false otherwise. *)
  let is_vector = foreign (ns "is_vector") (tensor @-> returning bool)

  (** [is_matrix tensor] checks if the tensor is a matrix (two non-unity dimensions).
      - [tensor] The tensor.
      - returns True if matrix, false otherwise. *)
  let is_matrix = foreign (ns "is_matrix") (tensor @-> returning bool)

  (** [is_3d tensor] checks if the tensor has exactly three non-unity dimensions.
      - [tensor] The tensor.
      - returns True if 3D, false otherwise. *)
  let is_3d = foreign (ns "is_3d") (tensor @-> returning bool)

  (** [n_dims tensor] returns the number of dimensions of the tensor (returns 1 for scalars).
      - [tensor] The tensor.
      - returns The number of dimensions. *)
  let n_dims = foreign (ns "n_dims") (tensor @-> returning int)

  (** [is_contiguous tensor] checks if the tensor's data is laid out contiguously in memory.
      - [tensor] The tensor.
      - returns True if contiguous, false otherwise. *)
  let is_contiguous = foreign (ns "is_contiguous") (tensor @-> returning bool)

  (** [is_contiguous_0 tensor] alias for `is_contiguous`.
      - [tensor] The tensor.
      - returns True if contiguous, false otherwise. *)
  let is_contiguous_0 = foreign (ns "is_contiguous_0") (tensor @-> returning bool)

  (** [is_contiguous_1 tensor] checks if the tensor is contiguous for dimensions >= 1.
      - [tensor] The tensor.
      - returns True if contiguous for dims >= 1, false otherwise. *)
  let is_contiguous_1 = foreign (ns "is_contiguous_1") (tensor @-> returning bool)

  (** [is_contiguous_2 tensor] checks if the tensor is contiguous for dimensions >= 2.
      - [tensor] The tensor.
      - returns True if contiguous for dims >= 2, false otherwise. *)
  let is_contiguous_2 = foreign (ns "is_contiguous_2") (tensor @-> returning bool)

  (** [is_contiguously_allocated tensor] returns whether the tensor elements are allocated as one contiguous block of
      memory (no gaps, but permutation ok).
      - [tensor] The tensor.
      - returns True if contiguously allocated, false otherwise. *)
  let is_contiguously_allocated = foreign (ns "is_contiguously_allocated") (tensor @-> returning bool)

  (** [is_contiguous_channels tensor] true for tensor that is stored in memory as CxWxHxN and has been permuted to
      WxHxCxN.
      - [tensor] The tensor.
      - returns True if contiguous channels, false otherwise. *)
  let is_contiguous_channels = foreign (ns "is_contiguous_channels") (tensor @-> returning bool)

  (** [is_contiguous_rows tensor] true if the elements in dimension 0 are contiguous, or there is just 1 block of
      elements.
      - [tensor] The tensor.
      - returns True if contiguous rows, false otherwise. *)
  let is_contiguous_rows = foreign (ns "is_contiguous_rows") (tensor @-> returning bool)

  (** [are_same_shape t0 t1] checks if two tensors have the same shape.
      - [t0] First tensor.
      - [t1] Second tensor.
      - returns True if shapes are identical, false otherwise. *)
  let are_same_shape = foreign (ns "are_same_shape") (tensor @-> tensor @-> returning bool)

  (** [are_same_stride t0 t1] checks if two tensors have the same strides.
      - [t0] First tensor.
      - [t1] Second tensor.
      - returns True if strides are identical, false otherwise. *)
  let are_same_stride = foreign (ns "are_same_stride") (tensor @-> tensor @-> returning bool)

  (** [can_repeat t0 t1] checks if tensor `t0` can be repeated (broadcasted) to match the shape of `t1`.
      - [t0] The tensor to potentially repeat.
      - [t1] The target shape tensor.
      - returns True if `t0` can be repeated to match `t1`, false otherwise. *)
  let can_repeat = foreign (ns "can_repeat") (tensor @-> tensor @-> returning bool)

  (** [tensor_overhead ()] returns the memory overhead of the ggml_tensor struct itself.
      - returns Overhead in bytes. *)
  let tensor_overhead = foreign (ns "tensor_overhead") (void @-> returning size_t)

  (** [validate_row_data typ data nbytes] validates if the data buffer is suitable for the given type and size.
      - [typ] The ggml type.
      - [data] Pointer to the data buffer.
      - [nbytes] Size of the data buffer in bytes.
      - returns True if the data is valid, false otherwise. *)
  let validate_row_data = foreign (ns "validate_row_data") (typ @-> ptr void @-> size_t @-> returning bool)

  (* Tensor Creation *)

  (** [new_tensor ctx typ n_dims ne] creates a new tensor with the specified type and dimensions.
      - [ctx] The context.
      - [typ] The tensor type.
      - [n_dims] The number of dimensions.
      - [ne] Pointer to an array containing the size of each dimension.
      - returns The new tensor. *)
  let new_tensor = foreign (ns "new_tensor") (context @-> typ @-> int @-> ptr int64_t @-> returning tensor)

  (** [new_tensor_1d ctx typ ne0] creates a new 1D tensor.
      - [ctx] The context.
      - [typ] The tensor type.
      - [ne0] Size of the first dimension.
      - returns The new tensor. *)
  let new_tensor_1d = foreign (ns "new_tensor_1d") (context @-> typ @-> int64_t @-> returning tensor)

  (** [new_tensor_2d ctx typ ne0 ne1] creates a new 2D tensor.
      - [ctx] The context.
      - [typ] The tensor type.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - returns The new tensor. *)
  let new_tensor_2d = foreign (ns "new_tensor_2d") (context @-> typ @-> int64_t @-> int64_t @-> returning tensor)

  (** [new_tensor_3d ctx typ ne0 ne1 ne2] creates a new 3D tensor.
      - [ctx] The context.
      - [typ] The tensor type.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [ne2] Size of the third dimension.
      - returns The new tensor. *)
  let new_tensor_3d =
    foreign (ns "new_tensor_3d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [new_tensor_4d ctx typ ne0 ne1 ne2 ne3] creates a new 4D tensor.
      - [ctx] The context.
      - [typ] The tensor type.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [ne2] Size of the third dimension.
      - [ne3] Size of the fourth dimension.
      - returns The new tensor. *)
  let new_tensor_4d =
    foreign (ns "new_tensor_4d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (* Buffer Creation *)

  (** [new_buffer ctx nbytes] allocates a buffer of the specified size within the context.
      - [ctx] The context.
      - [nbytes] The size of the buffer in bytes.
      - returns A pointer to the allocated buffer. *)
  let new_buffer = foreign (ns "new_buffer") (context @-> size_t @-> returning (ptr void))

  (* Tensor Duplication / Viewing *)

  (** [dup_tensor ctx src] duplicates a tensor, including its data.
      - [ctx] The context.
      - [src] The source tensor to duplicate.
      - returns The duplicated tensor. *)
  let dup_tensor = foreign (ns "dup_tensor") (context @-> tensor @-> returning tensor)

  (** [view_tensor ctx src] creates a view of a tensor. The view shares data with the original tensor.
      - [ctx] The context.
      - [src] The source tensor to view.
      - returns The view tensor. *)
  let view_tensor = foreign (ns "view_tensor") (context @-> tensor @-> returning tensor)

  (* Tensor Duplication / Viewing *)

  (** [aet_first_tensor ctx] returns the first tensor allocated in the context.
      - [ctx] The context.
      - returns The first tensor, or NULL if none exist. *)
  let get_first_tensor = foreign (ns "get_first_tensor") (context @-> returning tensor)

  (** [get_next_tensor ctx tensor] returns the next tensor allocated in the context after the given tensor.
      - [ctx] The context.
      - [tensor] The current tensor.
      - returns The next tensor, or NULL if it's the last one. *)
  let get_next_tensor = foreign (ns "get_next_tensor") (context @-> tensor @-> returning tensor)

  (** [get_tensor ctx name] retrieves a tensor from the context by its name.
      - [ctx] The context.
      - [name] The name of the tensor.
      - returns The tensor with the specified name, or NULL if not found. *)
  let get_tensor = foreign (ns "get_tensor") (context @-> string @-> returning tensor)

  (* Indexing *)

  (** [unravel_index tensor i i0 i1 i2 i3] converts a flat index `i` into multi-dimensional coordinates for the tensor.
      - [tensor] The tensor.
      - [i] The flat index.
      - [i0] Pointer to store the coordinate for the first dimension.
      - [i1] Pointer to store the coordinate for the second dimension.
      - [i2] Pointer to store the coordinate for the third dimension.
      - [i3] Pointer to store the coordinate for the fourth dimension. *)
  let unravel_index =
    foreign (ns "unravel_index")
      (tensor @-> int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> returning void)

  (* Op Info *)

  (** [get_unary_op tensor] returns the unary operation associated with the tensor, if any.
      - [tensor] The tensor.
      - returns The unary operation enum value. *)
  let get_unary_op = foreign (ns "get_unary_op") (tensor @-> returning unary_op)

  (** [get_glu_op tensor] returns the GLU operation associated with the tensor, if any.
      - [tensor] The tensor.
      - returns The GLU operation enum value. *)
  let get_glu_op = foreign (ns "get_glu_op") (tensor @-> returning glu_op)

  (* Data Access *)

  (** [get_data tensor] returns a raw pointer to the tensor's data.
      - [tensor] The tensor.
      - returns A void pointer to the data. *)
  let get_data = foreign (ns "get_data") (tensor @-> returning (ptr void))

  (** [get_data_f32 tensor] returns a pointer to the tensor's data, cast to float*.
      - [tensor] The tensor.
      - returns A float pointer to the data. *)
  let get_data_f32 = foreign (ns "get_data_f32") (tensor @-> returning (ptr float))

  (* Tensor Naming *)

  (** [get_name tensor] returns the name of the tensor.
      - [tensor] The tensor.
      - returns The name as a string. *)
  let get_name = foreign (ns "get_name") (tensor @-> returning string)

  (** [set_name tensor name] sets the name of the tensor.
      - [tensor] The tensor to name.
      - [name] The desired name.
      - returns The tensor itself. *)
  let set_name = foreign (ns "set_name") (tensor @-> string @-> returning tensor)
  (* ggml_format_name is variadic, skipping *)

  (* Tensor Flags *)

  (** [set_input tensor] marks the tensor as an input for the compute graph.
      - [tensor] The tensor to mark. *)
  let set_input = foreign (ns "set_input") (tensor @-> returning void)

  (** [set_output tensor] marks the tensor as an output for the compute graph.
      - [tensor] The tensor to mark. *)
  let set_output = foreign (ns "set_output") (tensor @-> returning void)

  (** [set_param ctx tensor] marks the tensor as containing trainable parameters.
      - [ctx] The context.
      - [tensor] The tensor to mark. *)
  let set_param = foreign (ns "set_param") (tensor @-> returning void)

  (** [set_loss tensor] marks the tensor as defining loss for numerical optimization. Multiple loss tensors add up.
      - [tensor] The tensor to mark. *)
  let set_loss = foreign (ns "set_loss") (tensor @-> returning void)

  (* Operations with backpropagation *)

  (** [dup ctx a] duplicates tensor `a`.
      - [ctx] The context.
      - [a] The tensor to duplicate.
      - returns The duplicated tensor. *)
  let dup = foreign (ns "dup") (context @-> tensor @-> returning tensor)

  (** [dup_inplace ctx a] duplicates tensor `a` in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] The tensor to duplicate.
      - returns A view of the tensor `a`. *)
  let dup_inplace = foreign (ns "dup_inplace") (context @-> tensor @-> returning tensor)

  (** [add ctx a b] computes `a + b`.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - returns The resulting tensor. *)
  let add = foreign (ns "add") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add_inplace ctx a b] computes `a + b` in-place, modifying `a`.
      - [ctx] The context.
      - [a] First tensor (modified).
      - [b] Second tensor.
      - returns The modified tensor `a`. *)
  let add_inplace = foreign (ns "add_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add_cast ctx a b typ] computes `a + b` and casts the result to `typ`.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - [typ] The target type for the result.
      - returns The resulting tensor. *)
  let add_cast = foreign (ns "add_cast") (context @-> tensor @-> tensor @-> typ @-> returning tensor)

  (** [add1 ctx a b] computes `a + b*1`. Adds the scalar `b` to each element of `a`.
      - [ctx] The context.
      - [a] The tensor.
      - [b] The scalar tensor to add.
      - returns The resulting tensor. *)
  let add1 = foreign (ns "add1") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add1_inplace ctx a b] computes `a + b*1` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - [b] The scalar tensor to add.
      - returns The modified tensor `a`. *)
  let add1_inplace = foreign (ns "add1_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [acc ctx a b nb1 nb2 nb3 offset] accumulates `b` into a view of `a`. `dst = a; view(dst, nb1, nb2, nb3, offset) +=
      b`.
      - [ctx] The context.
      - [a] The destination tensor.
      - [b] The tensor to accumulate.
      - [nb1] Stride for the first dimension of the view.
      - [nb2] Stride for the second dimension of the view.
      - [nb3] Stride for the third dimension of the view.
      - [offset] Offset in bytes for the view.
      - returns The modified tensor `a`. *)
  let acc =
    foreign (ns "acc") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [acc_inplace ctx a b nb1 nb2 nb3 offset] accumulates `b` into a view of `a` in-place.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The tensor to accumulate.
      - [nb1] Stride for the first dimension of the view.
      - [nb2] Stride for the second dimension of the view.
      - [nb3] Stride for the third dimension of the view.
      - [offset] Offset in bytes for the view.
      - returns The modified tensor `a`. *)
  let acc_inplace =
    foreign (ns "acc_inplace")
      (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [sub ctx a b] computes `a - b`.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - returns The resulting tensor. *)
  let sub = foreign (ns "sub") (context @-> tensor @-> tensor @-> returning tensor)

  (** [sub_inplace ctx a b] computes `a - b` in-place, modifying `a`.
      - [ctx] The context.
      - [a] First tensor (modified).
      - [b] Second tensor.
      - returns The modified tensor `a`. *)
  let sub_inplace = foreign (ns "sub_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul ctx a b] computes element-wise multiplication `a * b`.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - returns The resulting tensor. *)
  let mul = foreign (ns "mul") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul_inplace ctx a b] computes element-wise `a * b` in-place, modifying `a`.
      - [ctx] The context.
      - [a] First tensor (modified).
      - [b] Second tensor.
      - returns The modified tensor `a`. *)
  let mul_inplace = foreign (ns "mul_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [div ctx a b] computes element-wise division `a / b`.
      - [ctx] The context.
      - [a] Numerator tensor.
      - [b] Denominator tensor.
      - returns The resulting tensor. *)
  let div = foreign (ns "div") (context @-> tensor @-> tensor @-> returning tensor)

  (** [div_inplace ctx a b] computes element-wise `a / b` in-place, modifying `a`.
      - [ctx] The context.
      - [a] Numerator tensor (modified).
      - [b] Denominator tensor.
      - returns The modified tensor `a`. *)
  let div_inplace = foreign (ns "div_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [sqr ctx a] computes element-wise square `a^2`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let sqr = foreign (ns "sqr") (context @-> tensor @-> returning tensor)

  (** [sqr_inplace ctx a] computes element-wise `a^2` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let sqr_inplace = foreign (ns "sqr_inplace") (context @-> tensor @-> returning tensor)

  (** [sqrt ctx a] computes element-wise square root `sqrt(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let sqrt = foreign (ns "sqrt") (context @-> tensor @-> returning tensor)

  (** [sqrt_inplace ctx a] computes element-wise `sqrt(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let sqrt_inplace = foreign (ns "sqrt_inplace") (context @-> tensor @-> returning tensor)

  (** [log ctx a] computes element-wise natural logarithm `log(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let log = foreign (ns "log") (context @-> tensor @-> returning tensor)

  (** [log_inplace ctx a] computes element-wise `log(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let log_inplace = foreign (ns "log_inplace") (context @-> tensor @-> returning tensor)

  (** [sin ctx a] computes element-wise sine `sin(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let sin = foreign (ns "sin") (context @-> tensor @-> returning tensor)

  (** [sin_inplace ctx a] computes element-wise `sin(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let sin_inplace = foreign (ns "sin_inplace") (context @-> tensor @-> returning tensor)

  (** [cos ctx a] computes element-wise cosine `cos(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let cos = foreign (ns "cos") (context @-> tensor @-> returning tensor)

  (** [cos_inplace ctx a] computes element-wise `cos(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let cos_inplace = foreign (ns "cos_inplace") (context @-> tensor @-> returning tensor)

  (** [sum ctx a] computes the sum of all elements in `a`. Returns a scalar tensor.
      - [ctx] The context.
      - [a] The tensor.
      - returns A scalar tensor containing the sum. *)
  let sum = foreign (ns "sum") (context @-> tensor @-> returning tensor)

  (** [sum_rows ctx a] computes the sum along the first dimension (rows). Input shape [a,b,c,d] -> output shape
      [1,b,c,d].
      - [ctx] The context.
      - [a] The tensor.
      - returns A tensor containing the row sums. *)
  let sum_rows = foreign (ns "sum_rows") (context @-> tensor @-> returning tensor)

  (** [mean ctx a] computes the mean along the first dimension (rows).
      - [ctx] The context.
      - [a] The tensor.
      - returns A tensor containing the row means. *)
  let mean = foreign (ns "mean") (context @-> tensor @-> returning tensor)

  (** [argmax ctx a] computes the index of the maximum value along the first dimension (rows).
      - [ctx] The context.
      - [a] The tensor.
      - returns A tensor containing the indices of the maximum values. *)
  let argmax = foreign (ns "argmax") (context @-> tensor @-> returning tensor)

  (** [count_equal ctx a b] counts the number of equal elements between tensors `a` and `b`. Returns a scalar tensor.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - returns A scalar tensor containing the count. *)
  let count_equal = foreign (ns "count_equal") (context @-> tensor @-> tensor @-> returning tensor)

  (** [repeat ctx a b] repeats tensor `a` to match the shape of tensor `b`. If `a` already has the same shape as `b` and
      is not a parameter, `a` is returned directly.
      - [ctx] The context.
      - [a] The tensor to repeat.
      - [b] The tensor defining the target shape.
      - returns The repeated tensor. *)
  let repeat = foreign (ns "repeat") (context @-> tensor @-> tensor @-> returning tensor)

  (** [repeat_back ctx a b] sums repetitions in `a` back into the shape of `b`. This is the backward operation for
      `repeat`.
      - [ctx] The context.
      - [a] The tensor containing repetitions (gradient of `repeat` output).
      - [b] The tensor defining the target shape (original input to `repeat`).
      - returns The resulting tensor with summed repetitions. *)
  let repeat_back = foreign (ns "repeat_back") (context @-> tensor @-> tensor @-> returning tensor)

  (** [repeat_4d ctx a ne0 ne1 ne2 ne3] repeats tensor `a` to the specified shape.
      - [ctx] The context.
      - [a] The tensor to repeat.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [ne2] Size of the third dimension.
      - [ne3] Size of the fourth dimension.
      - returns The repeated tensor. *)
  let repeat_4d =
    foreign (ns "repeat_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [concat ctx a b dim] concatenates tensors `a` and `b` along the specified dimension `dim`.
      - [ctx] The context.
      - [a] First tensor.
      - [b] Second tensor.
      - [dim] The dimension along which to concatenate.
      - returns The concatenated tensor. *)
  let concat = foreign (ns "concat") (context @-> tensor @-> tensor @-> int @-> returning tensor)

  (** [abs ctx a] computes element-wise absolute value `abs(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let abs = foreign (ns "abs") (context @-> tensor @-> returning tensor)

  (** [abs_inplace ctx a] computes element-wise `abs(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let abs_inplace = foreign (ns "abs_inplace") (context @-> tensor @-> returning tensor)

  (** [sgn ctx a] computes element-wise sign `sgn(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let sgn = foreign (ns "sgn") (context @-> tensor @-> returning tensor)

  (** [sgn_inplace ctx a] computes element-wise `sgn(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let sgn_inplace = foreign (ns "sgn_inplace") (context @-> tensor @-> returning tensor)

  (** [neg ctx a] computes element-wise negation `-a`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let neg = foreign (ns "neg") (context @-> tensor @-> returning tensor)

  (** [neg_inplace ctx a] computes element-wise `-a` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let neg_inplace = foreign (ns "neg_inplace") (context @-> tensor @-> returning tensor)

  (** [step ctx a] computes element-wise step function (1 if x > 0, 0 otherwise).
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let step = foreign (ns "step") (context @-> tensor @-> returning tensor)

  (** [step_inplace ctx a] computes element-wise step function in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let step_inplace = foreign (ns "step_inplace") (context @-> tensor @-> returning tensor)

  (** [tanh ctx a] computes element-wise hyperbolic tangent `tanh(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let tanh = foreign (ns "tanh") (context @-> tensor @-> returning tensor)

  (** [tanh_inplace ctx a] computes element-wise `tanh(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let tanh_inplace = foreign (ns "tanh_inplace") (context @-> tensor @-> returning tensor)

  (** [elu ctx a] computes element-wise Exponential Linear Unit `elu(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let elu = foreign (ns "elu") (context @-> tensor @-> returning tensor)

  (** [elu_inplace ctx a] computes element-wise `elu(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let elu_inplace = foreign (ns "elu_inplace") (context @-> tensor @-> returning tensor)

  (** [relu ctx a] computes element-wise Rectified Linear Unit `relu(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let relu = foreign (ns "relu") (context @-> tensor @-> returning tensor)

  (** [leaky_relu ctx a negative_slope inplace] computes element-wise Leaky Rectified Linear Unit.
      - [ctx] The context.
      - [a] The tensor.
      - [negative_slope] The slope for negative values.
      - [inplace] Whether to perform the operation in-place.
      - returns The resulting tensor (or modified `a` if `inplace` is true). *)
  let leaky_relu = foreign (ns "leaky_relu") (context @-> tensor @-> float @-> bool @-> returning tensor)

  (** [relu_inplace ctx a] computes element-wise `relu(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let relu_inplace = foreign (ns "relu_inplace") (context @-> tensor @-> returning tensor)

  (** [sigmoid ctx a] computes element-wise sigmoid function `sigmoid(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let sigmoid = foreign (ns "sigmoid") (context @-> tensor @-> returning tensor)

  (** [sigmoid_inplace ctx a] computes element-wise `sigmoid(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let sigmoid_inplace = foreign (ns "sigmoid_inplace") (context @-> tensor @-> returning tensor)

  (** [gelu ctx a] computes element-wise Gaussian Error Linear Unit `gelu(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let gelu = foreign (ns "gelu") (context @-> tensor @-> returning tensor)

  (** [gelu_inplace ctx a] computes element-wise `gelu(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let gelu_inplace = foreign (ns "gelu_inplace") (context @-> tensor @-> returning tensor)

  (** [gelu_quick ctx a] computes element-wise approximate GELU `gelu_quick(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let gelu_quick = foreign (ns "gelu_quick") (context @-> tensor @-> returning tensor)

  (** [gelu_quick_inplace ctx a] computes element-wise `gelu_quick(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let gelu_quick_inplace = foreign (ns "gelu_quick_inplace") (context @-> tensor @-> returning tensor)

  (** [gelu_erf ctx a] computes GELU using erf (error function) when possible.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let gelu_erf = foreign (ns "gelu_erf") (context @-> tensor @-> returning tensor)

  (** [gelu_erf_inplace ctx a] computes GELU using erf in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let gelu_erf_inplace = foreign (ns "gelu_erf_inplace") (context @-> tensor @-> returning tensor)

  (** [silu ctx a] computes element-wise Sigmoid Linear Unit `silu(a) = a * sigmoid(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let silu = foreign (ns "silu") (context @-> tensor @-> returning tensor)

  (** [silu_inplace ctx a] computes element-wise `silu(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let silu_inplace = foreign (ns "silu_inplace") (context @-> tensor @-> returning tensor)

  (** [silu_back ctx a b] computes the backward pass for SiLU.
      - [ctx] The context.
      - [a] The input tensor `x` from the forward pass.
      - [b] The gradient `dy` from the output.
      - returns The gradient `dx`. *)
  let silu_back = foreign (ns "silu_back") (context @-> tensor @-> tensor @-> returning tensor)

  (** [hardswish ctx a] computes element-wise Hardswish `hardswish(a) = a * relu6(a + 3) / 6`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let hardswish = foreign (ns "hardswish") (context @-> tensor @-> returning tensor)

  (** [hardsigmoid ctx a] computes element-wise Hardsigmoid `hardsigmoid(a) = relu6(a + 3) / 6`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let hardsigmoid = foreign (ns "hardsigmoid") (context @-> tensor @-> returning tensor)

  (** [exp ctx a] computes element-wise exponentiation `exp(a)`.
      - [ctx] The context.
      - [a] The tensor.
      - returns The resulting tensor. *)
  let exp = foreign (ns "exp") (context @-> tensor @-> returning tensor)

  (** [exp_inplace ctx a] computes element-wise `exp(a)` in-place, modifying `a`.
      - [ctx] The context.
      - [a] The tensor (modified).
      - returns The modified tensor `a`. *)
  let exp_inplace = foreign (ns "exp_inplace") (context @-> tensor @-> returning tensor)

  (** [norm ctx a eps] normalizes `a` along the first dimension (rows).
      - [ctx] The context.
      - [a] The tensor to normalize.
      - [eps] Epsilon value for numerical stability.
      - returns The normalized tensor. *)
  let norm = foreign (ns "norm") (context @-> tensor @-> float @-> returning tensor)

  (** [norm_inplace ctx a eps] normalizes `a` along rows in-place.
      - [ctx] The context.
      - [a] The tensor to normalize (modified).
      - [eps] Epsilon value for numerical stability.
      - returns The modified tensor `a`. *)
  let norm_inplace = foreign (ns "norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm ctx a eps] computes Root Mean Square normalization along rows.
      - [ctx] The context.
      - [a] The tensor to normalize.
      - [eps] Epsilon value for numerical stability.
      - returns The normalized tensor. *)
  let rms_norm = foreign (ns "rms_norm") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm_inplace ctx a eps] computes RMS normalization along rows in-place.
      - [ctx] The context.
      - [a] The tensor to normalize (modified).
      - [eps] Epsilon value for numerical stability.
      - returns The modified tensor `a`. *)
  let rms_norm_inplace = foreign (ns "rms_norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [group_norm ctx a n_groups eps] computes Group Normalization. Normalizes along ne0*ne1*n_groups.
      - [ctx] The context.
      - [a] The tensor to normalize.
      - [n_groups] The number of groups.
      - [eps] Epsilon value for numerical stability.
      - returns The normalized tensor. *)
  let group_norm = foreign (ns "group_norm") (context @-> tensor @-> int @-> float @-> returning tensor)

  (** [group_norm_inplace ctx a n_groups eps] computes Group Normalization in-place.
      - [ctx] The context.
      - [a] The tensor to normalize (modified).
      - [n_groups] The number of groups.
      - [eps] Epsilon value for numerical stability.
      - returns The modified tensor `a`. *)
  let group_norm_inplace = foreign (ns "group_norm_inplace") (context @-> tensor @-> int @-> float @-> returning tensor)

  (** [l2_norm ctx a eps] computes L2 normalization along rows.
      - [ctx] The context.
      - [a] The tensor to normalize.
      - [eps] Epsilon value for numerical stability.
      - returns The normalized tensor. *)
  let l2_norm = foreign (ns "l2_norm") (context @-> tensor @-> float @-> returning tensor)

  (** [l2_norm_inplace ctx a eps] computes L2 normalization along rows in-place.
      - [ctx] The context.
      - [a] The tensor to normalize (modified).
      - [eps] Epsilon value for numerical stability.
      - returns The modified tensor `a`. *)
  let l2_norm_inplace = foreign (ns "l2_norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm_back ctx a b eps] computes the backward pass for RMS normalization.
      - [ctx] The context.
      - [a] The input tensor `x` from the forward pass.
      - [b] The gradient `dy` from the output.
      - [eps] Epsilon value used in the forward pass.
      - returns The gradient `dx`. *)
  let rms_norm_back = foreign (ns "rms_norm_back") (context @-> tensor @-> tensor @-> float @-> returning tensor)

  (** [mul_mat ctx a b] computes matrix multiplication `a * b^T`. A: [..., n, k], B: [..., m, k] -> Result: [..., m, n].
      - [ctx] The context.
      - [a] First matrix.
      - [b] Second matrix (transposed internally).
      - returns The resulting matrix. *)
  let mul_mat = foreign (ns "mul_mat") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul_mat_set_prec a prec] changes the precision used for the matrix multiplication involving tensor `a`.
      - [a] The tensor involved in the `mul_mat` operation (typically the output tensor).
      - [prec] The desired precision (e.g., `GGML_PREC_F32`). *)
  let mul_mat_set_prec = foreign (ns "mul_mat_set_prec") (tensor @-> prec @-> returning void)

  (** [mul_mat_id ctx as b ids] performs indirect matrix multiplication using IDs.
      - [ctx] The context.
      - [as] Tensor containing multiple matrices.
      - [b] The second matrix.
      - [ids] Tensor containing indices to select matrices from `as`.
      - returns The resulting matrix. *)
  let mul_mat_id = foreign (ns "mul_mat_id") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [out_prod ctx a b] computes the outer product of vectors `a` and `b`. A: [n, ...], B: [m, ...] -> Result:
      [m, n, ...].
      - [ctx] The context.
      - [a] First vector.
      - [b] Second vector.
      - returns The resulting matrix (outer product). *)
  let out_prod = foreign (ns "out_prod") (context @-> tensor @-> tensor @-> returning tensor)

  (** [scale ctx a s] scales tensor `a` by scalar `s`.
      - [ctx] The context.
      - [a] The tensor to scale.
      - [s] The scaling factor.
      - returns The scaled tensor. *)
  let scale = foreign (ns "scale") (context @-> tensor @-> float @-> returning tensor)

  (** [scale_inplace ctx a s] scales tensor `a` by scalar `s` in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] The tensor to scale (modified).
      - [s] The scaling factor.
      - returns A view of the modified tensor `a`. *)
  let scale_inplace = foreign (ns "scale_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [set ctx a b nb1 nb2 nb3 offset] sets the elements of a view of `a` to the values of `b`. Returns the modified
      `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The source tensor.
      - [nb1] Stride for the first dimension of the view.
      - [nb2] Stride for the second dimension of the view.
      - [nb3] Stride for the third dimension of the view.
      - [offset] Offset in bytes for the view.
      - returns The modified tensor `a`. *)
  let set =
    foreign (ns "set") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [set_inplace ctx a b nb1 nb2 nb3 offset] sets the elements of a view of `a` to the values of `b`. Returns a view
      of `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The source tensor.
      - [nb1] Stride for the first dimension of the view.
      - [nb2] Stride for the second dimension of the view.
      - [nb3] Stride for the third dimension of the view.
      - [offset] Offset in bytes for the view.
      - returns A view of the modified tensor `a`. *)
  let set_inplace =
    foreign (ns "set_inplace")
      (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [set_1d ctx a b offset] sets elements of `a` starting at `offset` to the values of 1D tensor `b`. Returns modified
      `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The 1D source tensor.
      - [offset] Offset in bytes.
      - returns The modified tensor `a`. *)
  let set_1d = foreign (ns "set_1d") (context @-> tensor @-> tensor @-> size_t @-> returning tensor)

  (** [set_1d_inplace ctx a b offset] sets elements of `a` starting at `offset` to the values of 1D tensor `b`. Returns
      a view of `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The 1D source tensor.
      - [offset] Offset in bytes.
      - returns A view of the modified tensor `a`. *)
  let set_1d_inplace = foreign (ns "set_1d_inplace") (context @-> tensor @-> tensor @-> size_t @-> returning tensor)

  (** [set_2d ctx a b nb1 offset] sets elements of a 2D view of `a` to the values of `b`. Returns modified `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The source tensor.
      - [nb1] Stride for the first dimension of the view.
      - [offset] Offset in bytes.
      - returns The modified tensor `a`. *)
  let set_2d = foreign (ns "set_2d") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> returning tensor)

  (** [set_2d_inplace ctx a b nb1 offset] sets elements of a 2D view of `a` to the values of `b`. Returns a view of `a`.
      - [ctx] The context.
      - [a] The destination tensor (modified).
      - [b] The source tensor.
      - [nb1] Stride for the first dimension of the view.
      - [offset] Offset in bytes.
      - returns A view of the modified tensor `a`. *)
  let set_2d_inplace =
    foreign (ns "set_2d_inplace") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> returning tensor)

  (** [cpy ctx a b] copies the data from tensor `a` to tensor `b`. Returns a view of `b`.
      - [ctx] The context.
      - [a] The source tensor.
      - [b] The destination tensor.
      - returns A view of the destination tensor `b`. *)
  let cpy = foreign (ns "cpy") (context @-> tensor @-> tensor @-> returning tensor)

  (** [cast ctx a typ] casts tensor `a` to the specified type `typ`.
      - [ctx] The context.
      - [a] The tensor to cast.
      - [typ] The target type.
      - returns The casted tensor. *)
  let cast = foreign (ns "cast") (context @-> tensor @-> typ @-> returning tensor)

  (** [cont ctx a] makes tensor `a` contiguous in memory.
      - [ctx] The context.
      - [a] The tensor.
      - returns A contiguous version of the tensor `a`. *)
  let cont = foreign (ns "cont") (context @-> tensor @-> returning tensor)

  (** [cont_1d ctx a ne0] makes tensor `a` contiguous with a new 1D shape.
      - [ctx] The context.
      - [a] The tensor.
      - [ne0] The size of the first dimension.
      - returns A contiguous 1D tensor. *)
  let cont_1d = foreign (ns "cont_1d") (context @-> tensor @-> int64_t @-> returning tensor)

  (** [cont_2d ctx a ne0 ne1] makes tensor `a` contiguous with a new 2D shape.
      - [ctx] The context.
      - [a] The tensor.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - returns A contiguous 2D tensor. *)
  let cont_2d = foreign (ns "cont_2d") (context @-> tensor @-> int64_t @-> int64_t @-> returning tensor)

  (** [cont_3d ctx a ne0 ne1 ne2] makes tensor `a` contiguous with a new 3D shape.
      - [ctx] The context.
      - [a] The tensor.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - [ne2] The size of the third dimension.
      - returns A contiguous 3D tensor. *)
  let cont_3d = foreign (ns "cont_3d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [cont_4d ctx a ne0 ne1 ne2 ne3] makes tensor `a` contiguous with a new 4D shape.
      - [ctx] The context.
      - [a] The tensor.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - [ne2] The size of the third dimension.
      - returns A contiguous 4D tensor. *)
  let cont_4d =
    foreign (ns "cont_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape ctx a b] creates a view of tensor `a` with the shape of tensor `b`.
      - [ctx] The context.
      - [a] The tensor to reshape.
      - [b] Tensor defining the new shape.
      - returns A view of `a` with the new shape. *)
  let reshape = foreign (ns "reshape") (context @-> tensor @-> tensor @-> returning tensor)

  (** [reshape_1d ctx a ne0] creates a 1D view of tensor `a`.
      - [ctx] The context.
      - [a] The tensor to reshape.
      - [ne0] The size of the first dimension.
      - returns A 1D view of `a`. *)
  let reshape_1d = foreign (ns "reshape_1d") (context @-> tensor @-> int64_t @-> returning tensor)

  (** [reshape_2d ctx a ne0 ne1] creates a 2D view of tensor `a`.
      - [ctx] The context.
      - [a] The tensor to reshape.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - returns A 2D view of `a`. *)
  let reshape_2d = foreign (ns "reshape_2d") (context @-> tensor @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape_3d ctx a ne0 ne1 ne2] creates a 3D view of tensor `a`.
      - [ctx] The context.
      - [a] The tensor to reshape.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - returns A 3D view of `a`. *)
  let reshape_3d =
    foreign (ns "reshape_3d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape_4d ctx a ne0 ne1 ne2 ne3] creates a 4D view of tensor `a`.
      - [ctx] The context.
      - [a] The tensor to reshape.
      - [ne0] The size of the first dimension.
      - [ne1] The size of the second dimension.
      - returns A 4D view of `a`. *)
  let reshape_4d =
    foreign (ns "reshape_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [view_1d ctx a ne0 offset] creates a 1D view of tensor `a` starting at `offset`.
      - [ctx] The context.
      - [a] The source tensor.
      - [ne0] The size of the view's dimension.
      - [offset] Offset in bytes from the start of `a`'s data.
      - returns The 1D view tensor. *)
  let view_1d = foreign (ns "view_1d") (context @-> tensor @-> int64_t @-> size_t @-> returning tensor)

  (** [view_2d ctx a ne0 ne1 nb1 offset] creates a 2D view of tensor `a`.
      - [ctx] The context.
      - [a] The source tensor.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [nb1] Row stride in bytes for the view.
      - [offset] Offset in bytes from the start of `a`'s data.
      - returns The 2D view tensor. *)
  let view_2d =
    foreign (ns "view_2d") (context @-> tensor @-> int64_t @-> int64_t @-> size_t @-> size_t @-> returning tensor)

  (** [view_3d ctx a ne0 ne1 ne2 nb1 nb2 offset] creates a 3D view of tensor `a`.
      - [ctx] The context.
      - [a] The source tensor.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [ne2] Size of the third dimension.
      - [nb1] Row stride in bytes.
      - [nb2] Slice stride in bytes.
      - [offset] Offset in bytes from the start of `a`'s data.
      - returns The 3D view tensor. *)
  let view_3d =
    foreign (ns "view_3d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [view_4d ctx a ne0 ne1 ne2 ne3 nb1 nb2 nb3 offset] creates a 4D view of tensor `a`.
      - [ctx] The context.
      - [a] The source tensor.
      - [ne0] Size of the first dimension.
      - [ne1] Size of the second dimension.
      - [ne2] Size of the third dimension.
      - [ne3] Size of the fourth dimension.
      - [nb1] Row stride in bytes.
      - [nb2] Slice stride in bytes.
      - [nb3] Stride for the fourth dimension in bytes.
      - [offset] Offset in bytes from the start of `a`'s data.
      - returns The 4D view tensor. *)
  let view_4d =
    foreign (ns "view_4d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> size_t
     @-> returning tensor)

  (** [permute ctx a axis0 axis1 axis2 axis3] permutes the dimensions of tensor `a`.
      - [ctx] The context.
      - [a] The tensor to permute.
      - [axis0] New index for the original dimension 0.
      - [axis1] New index for the original dimension 1.
      - [axis2] New index for the original dimension 2.
      - [axis3] New index for the original dimension 3.
      - returns The permuted tensor (view). *)
  let permute = foreign (ns "permute") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [transpose ctx a] transposes the first two dimensions of tensor `a`. Alias for `permute(ctx, a, 1, 0, 2, 3)`.
      - [ctx] The context.
      - [a] The tensor to transpose.
      - returns The transposed tensor (view). *)
  let transpose = foreign (ns "transpose") (context @-> tensor @-> returning tensor)

  (** [get_rows ctx a b] gathers rows from tensor `a` based on indices in tensor `b`. Supports 3D tensors where
      `a->ne[2] == b->ne[1]`.
      - [ctx] The context.
      - [a] The data tensor.
      - [b] The tensor containing row indices.
      - returns A tensor containing the gathered rows. *)
  let get_rows = foreign (ns "get_rows") (context @-> tensor @-> tensor @-> returning tensor)

  (** [get_rows_back ctx a b c] computes the backward pass for `get_rows`.
      - [ctx] The context.
      - [a] Gradient of the `get_rows` result.
      - [b] Row indices used in the forward pass.
      - [c] Original data tensor from the forward pass (used for shape).
      - returns The gradient with respect to the original data tensor `a`. *)
  let get_rows_back = foreign (ns "get_rows_back") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [set_rows ctx a b c] sets rows of tensor `a` with values from tensor `b` at indices specified by tensor `c`.
      - [ctx] The context.
      - [a] Destination tensor.
      - [b] Source tensor.
      - [c] Row indices.
      - returns The modified tensor `a`. *)
  let set_rows = foreign (ns "set_rows") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [diag ctx a] creates a diagonal matrix from vector `a`, or extracts the diagonal from matrix `a`.
      - [ctx] The context.
      - [a] The input tensor (vector or matrix).
      - returns The resulting diagonal matrix or vector. *)
  let diag = foreign (ns "diag") (context @-> tensor @-> returning tensor)

  (** [diag_mask_inf ctx a n_past] sets elements above the k-th diagonal (k = `n_past`) to -infinity.
      - [ctx] The context.
      - [a] The tensor to modify.
      - [n_past] The diagonal offset (0 for main diagonal, >0 for upper diagonals).
      - returns The modified tensor. *)
  let diag_mask_inf = foreign (ns "diag_mask_inf") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_inf_inplace ctx a n_past] sets elements above the k-th diagonal to -infinity in-place. Returns a view
      of `a`.
      - [ctx] The context.
      - [a] The tensor to modify (modified).
      - [n_past] The diagonal offset.
      - returns A view of the modified tensor `a`. *)
  let diag_mask_inf_inplace = foreign (ns "diag_mask_inf_inplace") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_zero ctx a n_past] sets elements above the k-th diagonal (k = `n_past`) to 0.
      - [ctx] The context.
      - [a] The tensor to modify.
      - [n_past] The diagonal offset.
      - returns The modified tensor. *)
  let diag_mask_zero = foreign (ns "diag_mask_zero") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_zero_inplace ctx a n_past] sets elements above the k-th diagonal to 0 in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] The tensor to modify (modified).
      - [n_past] The diagonal offset.
      - returns A view of the modified tensor `a`. *)
  let diag_mask_zero_inplace = foreign (ns "diag_mask_zero_inplace") (context @-> tensor @-> int @-> returning tensor)

  (** [soft_max ctx a] computes the softmax function along the first dimension (rows).
      - [ctx] The context.
      - [a] The input tensor.
      - returns The tensor with softmax applied. *)
  let soft_max = foreign (ns "soft_max") (context @-> tensor @-> returning tensor)

  (** [soft_max_inplace ctx a] computes softmax along rows in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] The input tensor (modified).
      - returns A view of the modified tensor `a`. *)
  let soft_max_inplace = foreign (ns "soft_max_inplace") (context @-> tensor @-> returning tensor)

  (** [soft_max_ext ctx a mask scale max_bias] computes fused softmax: `softmax(a*scale + mask*(ALiBi slope))`.
      - [ctx] The context.
      - [a] The input tensor.
      - [mask] Optional mask tensor.
      - [scale] Scaling factor for `a`.
      - [max_bias] Maximum bias for ALiBi (0.0f for no ALiBi).
      - returns The resulting tensor. *)
  let soft_max_ext = foreign (ns "soft_max_ext") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [soft_max_ext_back ctx a b scale max_bias] computes the backward pass for `soft_max_ext`.
      - [ctx] The context.
      - [a] Gradient of the `soft_max_ext` output.
      - [b] Original output of the `soft_max_ext` forward pass.
      - [scale] Scaling factor used in forward pass.
      - [max_bias] Maximum bias used in forward pass.
      - returns The gradient with respect to the input `a` of the forward pass. *)
  let soft_max_ext_back =
    foreign (ns "soft_max_ext_back") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [soft_max_ext_back_inplace ctx a b scale max_bias] computes the backward pass for `soft_max_ext` in-place. Returns
      a view of `a`.
      - [ctx] The context.
      - [a] Gradient tensor (modified).
      - [b] Original output of the `soft_max_ext` forward pass.
      - [scale] Scaling factor used in forward pass.
      - [max_bias] Maximum bias used in forward pass.
      - returns A view of the modified gradient tensor `a`. *)
  let soft_max_ext_back_inplace =
    foreign (ns "soft_max_ext_back_inplace") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [rope ctx a b n_dims mode] applies Rotary Positional Embedding (RoPE).
      - [ctx] The context.
      - [a] The input tensor.
      - [b] Tensor containing positions (int32, size a->ne[2]).
      - [n_dims] Number of dimensions to apply RoPE to.
      - [mode] RoPE mode flags (e.g., `GGML_ROPE_TYPE_NEOX`).
      - returns The tensor with RoPE applied. *)
  let rope = foreign (ns "rope") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [rope_inplace ctx a b n_dims mode] applies RoPE in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] The input tensor (modified).
      - [b] Tensor containing positions.
      - [n_dims] Number of dimensions for RoPE.
      - [mode] RoPE mode flags.
      - returns A view of the modified tensor `a`. *)
  let rope_inplace = foreign (ns "rope_inplace") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [rope_ext ctx a b c n_dims mode n_ctx_orig freq_base freq_scale ext_factor attn_factor beta_fast beta_slow]
      applies extended RoPE with custom parameters.
      - [ctx] The context.
      - [a] Input tensor.
      - [b] Positions tensor.
      - [c] Optional frequency factors tensor.
      - [n_dims] Number of dimensions for RoPE.
      - [mode] RoPE mode flags.
      - [n_ctx_orig] Original context size for scaling (e.g., YaRN).
      - [freq_base] Base frequency.
      - [freq_scale] Frequency scaling factor.
      - [ext_factor] Extrapolation factor (e.g., YaRN).
      - [attn_factor] Attention scaling factor (e.g., YaRN).
      - [beta_fast] Beta fast parameter (e.g., YaRN).
      - [beta_slow] Beta slow parameter (e.g., YaRN).
      - returns The tensor with extended RoPE applied. *)
  let rope_ext =
    foreign (ns "rope_ext")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_multi ctx a b c n_dims sections mode n_ctx_orig freq_base freq_scale ext_factor attn_factor beta_fast
       beta_slow] applies RoPE to multiple sections with different parameters.
      - [ctx] The context.
      - [a] Input tensor.
      - [b] Positions tensor.
      - [c] Optional frequency factors tensor.
      - [n_dims] Number of dimensions for RoPE.
      - [sections] Array defining the sections.
      - [mode] RoPE mode flags.
      - [n_ctx_orig] Original context size.
      - [freq_base] Base frequency.
      - [freq_scale] Frequency scaling factor.
      - [ext_factor] Extrapolation factor.
      - [attn_factor] Attention scaling factor.
      - [beta_fast] Beta fast parameter.
      - [beta_slow] Beta slow parameter.
      - returns The tensor with multi-section RoPE applied. *)
  let rope_multi =
    foreign (ns "rope_multi")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> ptr int @-> int @-> int @-> float @-> float @-> float
     @-> float @-> float @-> float @-> returning tensor)

  (** [rope_ext_inplace ctx a b c n_dims mode ...] applies extended RoPE in-place. Returns a view of `a`. (Parameters
      same as `rope_ext`).
      - returns A view of the modified tensor `a`. *)
  let rope_ext_inplace =
    foreign (ns "rope_ext_inplace")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_yarn_corr_dims n_dims n_ctx_orig freq_base beta_fast beta_slow dims] computes correction dimensions for YaRN
      RoPE scaling.
      - [n_dims] Number of dimensions for RoPE.
      - [n_ctx_orig] Original context size.
      - [freq_base] Base frequency.
      - [beta_fast] Beta fast parameter.
      - [beta_slow] Beta slow parameter.
      - returns Output pointer to store the two correction dimensions. *)
  let rope_yarn_corr_dims =
    foreign (ns "rope_yarn_corr_dims") (int @-> int @-> float @-> float @-> float @-> ptr float @-> returning void)

  (** [rope_ext_back ctx a b c n_dims mode ...] computes the backward pass for `rope_ext`. (Parameters mostly same as
      `rope_ext`, `a` is the gradient dy).
      - [a] Gradient of the `rope_ext` output.
      - [b] Positions tensor from forward pass.
      - [c] Optional frequency factors tensor from forward pass.
      - returns The gradient dx. *)
  let rope_ext_back =
    foreign (ns "rope_ext_back")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_multi_back ctx a b c n_dims sections mode ...] computes the backward pass for `rope_multi`. (Parameters
      mostly same as `rope_multi`, `a` is the gradient dy).
      - [a] Gradient of the `rope_multi` output.
      - [b] Positions tensor from forward pass.
      - [c] Optional frequency factors tensor from forward pass.
      - returns The gradient dx. *)
  let rope_multi_back =
    foreign (ns "rope_multi_back")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> ptr int @-> int @-> int @-> float @-> float @-> float
     @-> float @-> float @-> float @-> returning tensor)

  (** [clamp ctx a min max] clamps the elements of tensor `a` between `min` and `max`. Returns a view of `a`.
      - [ctx] The context.
      - [a] The tensor to clamp (modified).
      - [min] Minimum value.
      - [max] Maximum value.
      - returns A view of the modified tensor `a`. *)
  let clamp = foreign (ns "clamp") (context @-> tensor @-> float @-> float @-> returning tensor)

  (** [im2col ctx a b s0 s1 p0 p1 d0 d1 is_2D dst_type] implements the im2col operation used in convolutions.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0.
      - [p1] Padding dimension 1.
      - [d0] Dilation dimension 0.
      - [d1] Dilation dimension 1.
      - [is_2D] Whether it's a 2D operation.
      - [dst_type] The desired type for the output tensor.
      - returns The resulting tensor after im2col transformation. *)
  let im2col =
    foreign (ns "im2col")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> bool @-> typ
     @-> returning tensor)

  (** [im2col_back ctx a b ne s0 s1 p0 p1 d0 d1 is_2D] computes the backward pass for `im2col`.
      - [ctx] The context.
      - [a] Convolution kernel from forward pass.
      - [b] Gradient of the `im2col` output.
      - [ne] Shape of the original input data (`b` in forward pass).
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0.
      - [p1] Padding dimension 1.
      - [d0] Dilation dimension 0.
      - [d1] Dilation dimension 1.
      - [is_2D] Whether it was a 2D operation.
      - returns The gradient with respect to the input data. *)
  let im2col_back =
    foreign (ns "im2col_back")
      (context @-> tensor @-> tensor @-> ptr int64_t @-> int @-> int @-> int @-> int @-> int @-> int @-> bool
     @-> returning tensor)

  (** [conv_1d ctx a b s0 p0 d0] performs 1D convolution.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride.
      - [p0] Padding.
      - [d0] Dilation.
      - returns The result of the convolution. *)
  let conv_1d = foreign (ns "conv_1d") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_1d_ph ctx a b s d] performs 1D convolution with 'half' padding. Alias for `conv_1d(a, b, s, a->ne[0]/2, d)`.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s] Stride.
      - [d] Dilation.
      - returns The result of the convolution. *)
  let conv_1d_ph = foreign (ns "conv_1d_ph") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [conv_1d_dw ctx a b s0 p0 d0] performs 1D depthwise convolution.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride.
      - [p0] Padding.
      - [d0] Dilation.
      - returns The result of the depthwise convolution. *)
  let conv_1d_dw = foreign (ns "conv_1d_dw") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_1d_dw_ph ctx a b s0 d0] performs 1D depthwise convolution with 'half' padding.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride.
      - [d0] Dilation.
      - returns The result of the depthwise convolution. *)
  let conv_1d_dw_ph = foreign (ns "conv_1d_dw_ph") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [conv_transpose_1d ctx a b s0 p0 d0] performs 1D transposed convolution (deconvolution).
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride.
      - [p0] Padding.
      - [d0] Dilation.
      - returns The result of the transposed convolution. *)
  let conv_transpose_1d =
    foreign (ns "conv_transpose_1d") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_2d_direct ctx a b s0 s1 p0 p1 d0 d1] performs 2D convolution directly.
      - [ctx] The context.
      - [a] Convolution kernel [KW, KH, IC, OC].
      - [b] Input data [W, H, C, N].
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0.
      - [p1] Padding dimension 1.
      - [d0] Dilation dimension 0.
      - [d1] Dilation dimension 1.
      - returns The result of the 2D convolution. *)
  let conv_2d_direct =
    foreign (ns "conv_2d_direct")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [conv_2d_sk_p0 ctx a b] performs 2D convolution with stride equal to kernel size and zero padding.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - returns The result of the convolution. *)
  let conv_2d_sk_p0 = foreign (ns "conv_2d_sk_p0") (context @-> tensor @-> tensor @-> returning tensor)

  (** [conv_2d_s1_ph ctx a b] performs 2D convolution with stride 1 and 'half' padding.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - returns The result of the convolution. *)
  let conv_2d_s1_ph = foreign (ns "conv_2d_s1_ph") (context @-> tensor @-> tensor @-> returning tensor)

  (** [conv_2d_dw ctx a b s0 s1 p0 p1 d0 d1] performs 2D depthwise convolution.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0.
      - [p1] Padding dimension 1.
      - [d0] Dilation dimension 0.
      - [d1] Dilation dimension 1.
      - returns The result of the 2D depthwise convolution. *)
  let conv_2d_dw =
    foreign (ns "conv_2d_dw")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [conv_transpose_2d_p0 ctx a b stride] performs 2D transposed convolution with zero padding.
      - [ctx] The context.
      - [a] Convolution kernel.
      - [b] Input data.
      - [stride] The stride.
      - returns The result of the transposed convolution. *)
  let conv_transpose_2d_p0 =
    foreign (ns "conv_transpose_2d_p0") (context @-> tensor @-> tensor @-> int @-> returning tensor)

  (** [pool_1d ctx a op k0 s0 p0] performs 1D pooling.
      - [ctx] The context.
      - [a] Input tensor.
      - [op] Pooling operation type (`GGML_OP_POOL_MAX`, `GGML_OP_POOL_AVG`).
      - [k0] Kernel size.
      - [s0] Stride.
      - [p0] Padding.
      - returns The result of the 1D pooling. *)
  let pool_1d = foreign (ns "pool_1d") (context @-> tensor @-> op_pool @-> int @-> int @-> int @-> returning tensor)

  (** [pool_2d ctx a op k0 k1 s0 s1 p0 p1] performs 2D pooling.
      - [ctx] The context.
      - [a] Input tensor.
      - [op] Pooling operation type.
      - [k0] Kernel size dimension 0.
      - [k1] Kernel size dimension 1.
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0 (float for potential fractional padding).
      - [p1] Padding dimension 1 (float for potential fractional padding).
      - returns The result of the 2D pooling. *)
  let pool_2d =
    foreign (ns "pool_2d")
      (context @-> tensor @-> op_pool @-> int @-> int @-> int @-> int @-> float @-> float @-> returning tensor)

  (** [pool_2d_back ctx a af op k0 k1 s0 s1 p0 p1] computes the backward pass for 2D pooling.
      - [ctx] The context.
      - [a] Gradient of the `pool_2d` output.
      - [af] Original input tensor from the forward pass.
      - [op] Pooling operation type used in forward pass.
      - [k0] Kernel size dimension 0.
      - [k1] Kernel size dimension 1.
      - [s0] Stride dimension 0.
      - [s1] Stride dimension 1.
      - [p0] Padding dimension 0.
      - [p1] Padding dimension 1.
      - returns The gradient with respect to the input of the forward pass. *)
  let pool_2d_back =
    foreign (ns "pool_2d_back")
      (context @-> tensor @-> tensor @-> op_pool @-> int @-> int @-> int @-> int @-> float @-> float
     @-> returning tensor)

  (** [interpolate ctx a ne0 ne1 ne2 ne3 mode] up- or downsamples the input to the specified size. 2D scale modes (e.g.,
      bilinear) are applied to the first two dimensions.
      - [ctx] The context.
      - [a] Input tensor.
      - [ne0] Target size for dimension 0.
      - [ne1] Target size for dimension 1.
      - [ne2] Target size for dimension 2.
      - [ne3] Target size for dimension 3.
      - [mode] The scaling mode (ggml_scale_mode | ggml_scale_flag...).
      - returns The interpolated tensor. *)
  let interpolate =
    foreign (ns "interpolate")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> uint32_t @-> returning tensor)

  (** [pad ctx a p0 p1 p2 p3] pads each dimension of tensor `a` with zeros.
      - [ctx] The context.
      - [a] Input tensor.
      - [p0] Padding for dimension 0.
      - [p1] Padding for dimension 1.
      - [p2] Padding for dimension 2.
      - [p3] Padding for dimension 3.
      - returns The padded tensor. *)
  let pad = foreign (ns "pad") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [pad_reflect_1d ctx a p0 p1] pads the first two dimensions of tensor `a` using reflection padding.
      - [ctx] The context.
      - [a] Input tensor.
      - [p0] Padding for dimension 0.
      - [p1] Padding for dimension 1.
      - returns The padded tensor. *)
  let pad_reflect_1d = foreign (ns "pad_reflect_1d") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [roll ctx a shift0 shift1 shift2 shift3] moves tensor elements by an offset given for each dimension. Elements
      that are shifted beyond the last position are wrapped around to the beginning.
      - [ctx] The context.
      - [a] The tensor to roll.
      - [shift0] Shift for dimension 0.
      - [shift1] Shift for dimension 1.
      - [shift2] Shift for dimension 2.
      - [shift3] Shift for dimension 3.
      - returns The rolled tensor. *)
  let roll = foreign (ns "roll") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [timestep_embedding ctx timesteps dim max_period] creates timestep embeddings used in diffusion models.
      - [ctx] The context.
      - [timesteps] Tensor of timesteps [N,].
      - [dim] Embedding dimension.
      - [max_period] Maximum period for the sinusoidal embedding.
      - returns Tensor of embeddings [N, dim]. *)
  let timestep_embedding = foreign (ns "timestep_embedding") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [argsort ctx a order] returns the indices that would sort tensor `a` along rows.
      - [ctx] The context.
      - [a] Input tensor.
      - [order] Sort order (`GGML_SORT_ORDER_ASC` or `GGML_SORT_ORDER_DESC`).
      - returns Tensor containing the sorted indices. *)
  let argsort = foreign (ns "argsort") (context @-> tensor @-> sort_order @-> returning tensor)

  (** [arange ctx start stop step] creates a 1D tensor with values ranging from `start` to `stop` (exclusive) with
      `step`.
      - [ctx] The context.
      - [start] Start value.
      - [stop] Stop value.
      - [step] Step value.
      - returns The 1D tensor containing the range. *)
  let arange = foreign (ns "arange") (context @-> float @-> float @-> float @-> returning tensor)

  (** [top_k ctx a k] returns the values and indices of the top `k` elements along the last dimension.
      - [ctx] The context.
      - [a] Input tensor.
      - [k] The number of top elements to select.
      - returns A tensor containing the top k values and indices (implementation specific). *)
  let top_k = foreign (ns "top_k") (context @-> tensor @-> int @-> returning tensor)

  (** [flash_attn_ext ctx q k v mask scale max_bias logit_softcap] performs extended Flash Attention. q:
      [n_embd_k, n_batch, n_head, 1], k: [n_embd_k, n_kv, n_head_kv, 1], v: [n_embd_v, n_kv, n_head_kv, 1], mask:
      [n_kv, n_batch_pad, 1, 1]
      - [ctx] The context.
      - [q] Query tensor.
      - [k] Key tensor.
      - [v] Value tensor (not transposed).
      - [mask] Optional attention mask.
      - [scale] Scaling factor for QK^T
      - [max_bias] Maximum bias for ALiBi.
      - [logit_softcap] Softcap value for logits.
      - returns Result tensor [n_embd_v, n_head, n_batch, 1] (permuted). *)
  let flash_attn_ext =
    foreign (ns "flash_attn_ext")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> float @-> float @-> float @-> returning tensor)

  (** [flash_attn_ext_set_prec a prec] sets the precision for the Flash Attention operation involving tensor `a`.
      - [a] The tensor involved in the Flash Attention operation.
      - [prec] The desired precision. *)
  let flash_attn_ext_set_prec = foreign (ns "flash_attn_ext_set_prec") (tensor @-> prec @-> returning void)

  (** [flash_attn_ext_get_prec a] gets the precision currently set for the Flash Attention operation involving tensor
      `a`.
      - [a] The tensor involved in the Flash Attention operation.
      - returns The current precision. *)
  let flash_attn_ext_get_prec = foreign (ns "flash_attn_ext_get_prec") (tensor @-> returning prec)

  (** [flash_attn_back ctx q k v d masked] computes the backward pass for Flash Attention. (Note: Needs adaptation for
      `flash_attn_ext`).
      - [ctx] The context.
      - [q] Query tensor from forward pass.
      - [k] Key tensor from forward pass.
      - [v] Value tensor from forward pass.
      - [d] Gradient of the Flash Attention output.
      - [masked] Whether masking was used in the forward pass.
      - returns Gradient with respect to the input(s). *)
  let flash_attn_back =
    foreign (ns "flash_attn_back") (context @-> tensor @-> tensor @-> tensor @-> tensor @-> bool @-> returning tensor)

  (** [ssm_conv ctx sx c] performs Structured State Space Model (SSM) convolution.
      - [ctx] The context.
      - [sx] State tensor.
      - [c] Convolution kernel.
      - returns Result of the SSM convolution. *)
  let ssm_conv = foreign (ns "ssm_conv") (context @-> tensor @-> tensor @-> returning tensor)

  (** [ssm_scan ctx s x dt A B C] performs Structured State Space Model (SSM) scan.
      - [ctx] The context.
      - [s] State tensor.
      - [x] Input tensor.
      - [dt] Delta t tensor.
      - [A] State transition matrix A.
      - [B] State transition matrix B.
      - [C] Output matrix C.
      - returns Result of the SSM scan. *)
  let ssm_scan =
    foreign (ns "ssm_scan")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [win_part ctx a w] partitions tensor `a` into non-overlapping windows of size `w`.
      - [ctx] The context.
      - [a] Input tensor.
      - [w] Window size.
      - returns Tensor containing the window partitions. *)
  let win_part = foreign (ns "win_part") (context @-> tensor @-> int @-> returning tensor)

  (** [win_unpart ctx a w0 h0 w] reverses the window partitioning operation.
      - [ctx] The context.
      - [a] Tensor containing window partitions.
      - [w0] Original width before partitioning.
      - [h0] Original height before partitioning.
      - [w] Window size used during partitioning.
      - returns The reconstructed tensor. *)
  let win_unpart = foreign (ns "win_unpart") (context @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [unary ctx a op] applies a unary operation `op` to tensor `a`.
      - [ctx] The context.
      - [a] Input tensor.
      - [op] Unary operation type.
      - returns The resulting tensor. *)
  let unary = foreign (ns "unary") (context @-> tensor @-> unary_op @-> returning tensor)

  (** [unary_inplace ctx a op] applies a unary operation `op` to tensor `a` in-place.
      - [ctx] The context.
      - [a] Input tensor (modified).
      - [op] Unary operation type.
      - returns The modified tensor `a`. *)
  let unary_inplace = foreign (ns "unary_inplace") (context @-> tensor @-> unary_op @-> returning tensor)

  (** [get_rel_pos ctx a qh kh] computes relative positional embeddings. Used in SAM.
      - [ctx] The context.
      - [a] Input tensor containing positional information.
      - [qh] Query height/width.
      - [kh] Key height/width.
      - returns Tensor containing relative positional embeddings. *)
  let get_rel_pos = foreign (ns "get_rel_pos") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [add_rel_pos ctx a pw ph] adds relative positional embeddings to tensor `a`. Used in SAM.
      - [ctx] The context.
      - [a] Input tensor (e.g., attention scores).
      - [pw] Relative position embedding for width.
      - [ph] Relative position embedding for height.
      - returns Tensor with added positional embeddings. *)
  let add_rel_pos = foreign (ns "add_rel_pos") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [add_rel_pos_inplace ctx a pw ph] adds relative positional embeddings to `a` in-place. Returns a view of `a`.
      - [ctx] The context.
      - [a] Input tensor (modified).
      - [pw] Relative position embedding for width.
      - [ph] Relative position embedding for height.
      - returns A view of the modified tensor `a`. *)
  let add_rel_pos_inplace =
    foreign (ns "add_rel_pos_inplace") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [rwkv_wkv6 ctx k v r tf td state] computes the RWKV v6 WKV operation.
      - [ctx] The context.
      - [k] Key tensor.
      - [v] Value tensor.
      - [r] Receptance tensor.
      - [tf] Time factor tensor.
      - [td] Time decay tensor.
      - [state] State tensor.
      - returns Result of the WKV operation. *)
  let rwkv_wkv6 =
    foreign (ns "rwkv_wkv6")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [gated_linear_attn ctx k v q g state scale] computes Gated Linear Attention.
      - [ctx] The context.
      - [k] Key tensor.
      - [v] Value tensor.
      - [q] Query tensor.
      - [g] Gate tensor.
      - [state] State tensor.
      - [scale] Scaling factor.
      - returns Result of the Gated Linear Attention. *)
  let gated_linear_attn =
    foreign (ns "gated_linear_attn")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> float @-> returning tensor)

  (** [rwkv_wkv7 ctx r w k v a b state] computes the RWKV v7 WKV operation.
      - [ctx] The context.
      - [r] Receptance tensor.
      - [w] Weight tensor.
      - [k] Key tensor.
      - [v] Value tensor.
      - [a] Alpha tensor (state).
      - [b] Beta tensor (state).
      - [state] State tensor (previous state).
      - returns Result of the WKV operation. *)
  let rwkv_wkv7 =
    foreign (ns "rwkv_wkv7")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [map_custom1 ctx a fun n_tasks userdata] applies a custom unary function `fun` to tensor `a`.
      - [ctx] The context.
      - [a] Input tensor.
      - [fun] The custom function `(dst, a, ith, nth, userdata) -> void`.
      - [n_tasks] Number of parallel tasks to use (-1 for max).
      - [userdata] User data passed to the function.
      - returns The resulting tensor. *)
  let map_custom1 =
    foreign (ns "map_custom1") (context @-> tensor @-> custom1_op_t @-> int @-> ptr void @-> returning tensor)

  (** [map_custom1_inplace ctx a fun n_tasks userdata] applies a custom unary function `fun` to tensor `a` in-place.
      - [ctx] The context.
      - [a] Input tensor (modified).
      - [fun] The custom function.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The modified tensor `a`. *)
  let map_custom1_inplace =
    foreign (ns "map_custom1_inplace") (context @-> tensor @-> custom1_op_t @-> int @-> ptr void @-> returning tensor)

  (** [map_custom2 ctx a b fun n_tasks userdata] applies a custom binary function `fun` to tensors `a` and `b`.
      - [ctx] The context.
      - [a] First input tensor.
      - [b] Second input tensor.
      - [fun] The custom function `(dst, a, b, ith, nth, userdata) -> void`.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The resulting tensor. *)
  let map_custom2 =
    foreign (ns "map_custom2") (context @-> tensor @-> tensor @-> custom2_op_t @-> int @-> ptr void @-> returning tensor)

  (** [map_custom2_inplace ctx a b fun n_tasks userdata] applies a custom binary function `fun` to `a` and `b` in-place
      (modifies `a`).
      - [ctx] The context.
      - [a] First input tensor (modified).
      - [b] Second input tensor.
      - [fun] The custom function.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The modified tensor `a`. *)
  let map_custom2_inplace =
    foreign (ns "map_custom2_inplace")
      (context @-> tensor @-> tensor @-> custom2_op_t @-> int @-> ptr void @-> returning tensor)

  (** [map_custom3 ctx a b c fun n_tasks userdata] applies a custom ternary function `fun` to tensors `a`, `b`, and `c`.
      - [ctx] The context.
      - [a] First input tensor.
      - [b] Second input tensor.
      - [c] Third input tensor.
      - [fun] The custom function `(dst, a, b, c, ith, nth, userdata) -> void`.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The resulting tensor. *)
  let map_custom3 =
    foreign (ns "map_custom3")
      (context @-> tensor @-> tensor @-> tensor @-> custom3_op_t @-> int @-> ptr void @-> returning tensor)

  (** [map_custom3_inplace ctx a b c fun n_tasks userdata] applies a custom ternary function `fun` to `a`, `b`, and `c`
      in-place (modifies `a`).
      - [ctx] The context.
      - [a] First input tensor (modified).
      - [b] Second input tensor.
      - [c] Third input tensor.
      - [fun] The custom function.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The modified tensor `a`. *)
  let map_custom3_inplace =
    foreign (ns "map_custom3_inplace")
      (context @-> tensor @-> tensor @-> tensor @-> custom3_op_t @-> int @-> ptr void @-> returning tensor)

  (** [custom_4d ctx typ ne0 ne1 ne2 ne3 args n_args fun n_tasks userdata] creates a tensor using a custom operation
      with multiple arguments.
      - [ctx] The context.
      - [typ] The type of the resulting tensor.
      - [ne0] Size of dimension 0.
      - [ne1] Size of dimension 1.
      - [ne2] Size of dimension 2.
      - [ne3] Size of dimension 3.
      - [args] Pointer to an array of input tensor pointers.
      - [n_args] Number of input tensors in `args`.
      - [fun] The custom function `(dst, ith, nth, userdata) -> void`.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The resulting tensor. *)
  let custom_4d =
    foreign (ns "custom_4d")
      (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> ptr tensor @-> int @-> custom_op_t @-> int
     @-> ptr void @-> returning tensor)

  (** [custom_inplace ctx a args n_args fun n_tasks userdata] applies a custom operation with multiple arguments
      in-place (modifies `a`).
      - [ctx] The context.
      - [a] The tensor to modify.
      - [args] Pointer to an array of input tensor pointers.
      - [n_args] Number of input tensors in `args`.
      - [fun] The custom function `(dst, ith, nth, userdata) -> void`.
      - [n_tasks] Number of parallel tasks.
      - [userdata] User data.
      - returns The modified tensor `a`. *)
  let custom_inplace =
    foreign (ns "custom_inplace")
      (context @-> tensor @-> ptr tensor @-> int @-> custom_op_t @-> int @-> ptr void @-> returning tensor)

  (** [quantize_init typ] initializes quantization resources for the given type.
      - [typ] The quantization type. *)
  let quantize_init = foreign (ns "quantize_init") (typ @-> returning void)

  (** [quantize_free ()] frees quantization resources. *)
  let quantize_free = foreign (ns "quantize_free") (void @-> returning void)

  (** [quantize_requires_imatrix typ] checks if the quantization type requires an importance matrix.
      - [typ] The quantization type.
      - returns True if an importance matrix is required, false otherwise. *)
  let quantize_requires_imatrix = foreign (ns "quantize_requires_imatrix") (typ @-> returning bool)

  (** [quantize_chunk typ src dst start N num_threads imatrix] quantizes a chunk of data.
      - [typ] Target quantization type.
      - [src] Pointer to the source f32 data.
      - [dst] Pointer to the destination quantized data buffer.
      - [start] Starting index of the chunk.
      - [N] Number of elements in the chunk.
      - [num_threads] Number of threads to use (unused in current C impl).
      - [imatrix] Optional importance matrix.
      - returns Size of the quantized data in bytes. *)
  let quantize_chunk =
    foreign (ns "quantize_chunk")
      (typ @-> ptr float @-> ptr void @-> int64_t @-> int64_t @-> int64_t @-> ptr float @-> returning size_t)

  (** [cross_entropy_loss ctx a b] computes the cross-entropy loss between `a` (logits) and `b` (labels).
      - [ctx] The context.
      - [a] Logits tensor.
      - [b] Labels tensor.
      - returns Scalar tensor containing the loss. *)
  let cross_entropy_loss = foreign (ns "cross_entropy_loss") (context @-> tensor @-> tensor @-> returning tensor)

  (** [cross_entropy_loss_back ctx a b c] computes the backward pass for cross-entropy loss.
      - [ctx] The context.
      - [a] Gradient of the loss.
      - [b] Logits tensor from the forward pass.
      - [c] Labels tensor from the forward pass.
      - returns Gradient with respect to the logits `a`. *)
  let cross_entropy_loss_back =
    foreign (ns "cross_entropy_loss_back") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [opt_step_adamw ctx w dw m v hparams] performs an AdamW optimization step.
      - [ctx] The context.
      - [w] Weight tensor (modified).
      - [dw] Gradient tensor.
      - [m] First moment tensor (modified).
      - [v] Second moment tensor (modified).
      - [hparams] Hyperparameters tensor.
      - returns The modified weight tensor `w`. *)
  let opt_step_adamw =
    foreign (ns "opt_step_adamw") (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [build_forward_expand graph tensor] expands the forward graph to include the computation of `tensor`.
      - [graph] The computation graph.
      - [tensor] The tensor whose computation to include. *)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> tensor @-> returning void)

  (** [build_backward_expand ctx cgraph grad_accs] expands the backward graph.
      - [ctx] Context for gradient computation.
      - [cgraph] The compute graph.
      - [grad_accs] Pointer to an array of gradient accumulation tensors. *)
  let build_backward_expand = foreign (ns "build_backward_expand") (context @-> cgraph @-> ptr tensor @-> returning void)

  (** [glu ctx a op swapped] performs a Gated Linear Unit operation.
      - [ctx] The context.
      - [a] Input tensor (n columns, r rows).
      - [op] The GLU operation type.
      - [swapped] If true, expects gate in the first half of the row.
      - returns Resulting tensor (n/2 columns, r rows). *)
  let glu = foreign (ns "glu") (context @-> tensor @-> glu_op @-> bool @-> returning tensor)

  (** [reglu ctx a] performs a ReGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let reglu = foreign (ns "reglu") (context @-> tensor @-> returning tensor)

  (** [reglu_swapped ctx a] performs a swapped ReGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let reglu_swapped = foreign (ns "reglu_swapped") (context @-> tensor @-> returning tensor)

  (** [geglu ctx a] performs a GeGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let geglu = foreign (ns "geglu") (context @-> tensor @-> returning tensor)

  (** [geglu_swapped ctx a] performs a swapped GeGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let geglu_swapped = foreign (ns "geglu_swapped") (context @-> tensor @-> returning tensor)

  (** [swiglu ctx a] performs a SwiGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let swiglu = foreign (ns "swiglu") (context @-> tensor @-> returning tensor)

  (** [swiglu_swapped ctx a] performs a swapped SwiGLU operation.
      - [ctx] The context.
      - [a] Input tensor.
      - returns Resulting tensor. *)
  let swiglu_swapped = foreign (ns "swiglu_swapped") (context @-> tensor @-> returning tensor)

  (** [glu_split ctx a b op] performs a Gated Linear Unit operation with split inputs.
      - [ctx] The context.
      - [a] First input tensor (n columns, r rows).
      - [b] Second input tensor (n columns, r rows).
      - [op] The GLU operation type.
      - returns Resulting tensor. *)
  let glu_split = foreign (ns "glu_split") (context @-> tensor @-> tensor @-> glu_op @-> returning tensor)

  (** [reglu_split ctx a b] performs a ReGLU operation with split inputs.
      - [ctx] The context.
      - [a] First input tensor.
      - [b] Second input tensor.
      - returns Resulting tensor. *)
  let reglu_split = foreign (ns "reglu_split") (context @-> tensor @-> tensor @-> returning tensor)

  (** [geglu_split ctx a b] performs a GeGLU operation with split inputs.
      - [ctx] The context.
      - [a] First input tensor.
      - [b] Second input tensor.
      - returns Resulting tensor. *)
  let geglu_split = foreign (ns "geglu_split") (context @-> tensor @-> tensor @-> returning tensor)

  (** [swiglu_split ctx a b] performs a SwiGLU operation with split inputs.
      - [ctx] The context.
      - [a] First input tensor.
      - [b] Second input tensor.
      - returns Resulting tensor. *)
  let swiglu_split = foreign (ns "swiglu_split") (context @-> tensor @-> tensor @-> returning tensor)

  (** [new_graph ctx] creates a new computation graph with the default size.
      - [ctx] The context.
      - returns The new computation graph. *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)

  (** [new_graph_custom ctx size grads] creates a new computation graph with a custom size.
      - [ctx] The context.
      - [size] The maximum number of nodes in the graph.
      - [grads] Whether the graph will store gradients.
      - returns The new computation graph. *)
  let new_graph_custom = foreign (ns "new_graph_custom") (context @-> size_t @-> bool @-> returning cgraph)

  (** [graph_dup ctx cgraph force_grads] duplicates a computation graph.
      - [ctx] The context.
      - [cgraph] The graph to duplicate.
      - [force_grads] Whether to force gradient storage in the duplicated graph.
      - returns The duplicated graph. *)
  let graph_dup = foreign (ns "graph_dup") (context @-> cgraph @-> bool @-> returning cgraph)

  (** [graph_cpy src dst] copies the nodes from graph `src` to `dst`.
      - [src] The source graph.
      - [dst] The destination graph. *)
  let graph_cpy = foreign (ns "graph_cpy") (cgraph @-> cgraph @-> returning void)

  (** [graph_reset graph] resets the gradient data for all nodes in the graph.
      - [graph] The computation graph. *)
  let graph_reset = foreign (ns "graph_reset") (cgraph @-> returning void)

  (** [graph_clear graph] clears the nodes from the graph.
      - [graph] The computation graph. *)
  let graph_clear = foreign (ns "graph_clear") (cgraph @-> returning void)

  (** [graph_size graph] returns the number of nodes currently in the graph.
      - [graph] The computation graph.
      - returns The number of nodes. *)
  let graph_size = foreign (ns "graph_size") (cgraph @-> returning int)

  (** [graph_node graph i] returns the i-th tensor node in the graph.
      - [graph] The computation graph.
      - [i] The index of the node.
      - returns The tensor node. *)
  let graph_node = foreign (ns "graph_node") (cgraph @-> int @-> returning tensor)

  (** [graph_nodes graph] returns a pointer to the array of tensor nodes in the graph.
      - [graph] The computation graph.
      - returns Pointer to the first tensor node. *)
  let graph_nodes = foreign (ns "graph_nodes") (cgraph @-> returning (ptr tensor))
  (* Returns ptr to the first tensor *)

  (** [graph_n_nodes graph] returns the number of nodes currently in the graph (same as `graph_size`).
      - [graph] The computation graph.
      - returns The number of nodes. *)
  let graph_n_nodes = foreign (ns "graph_n_nodes") (cgraph @-> returning int)

  (** [graph_add_node graph tensor] adds a tensor node to the graph. (Internal use likely).
      - [graph] The computation graph.
      - [tensor] The tensor node to add. *)
  let graph_add_node = foreign (ns "graph_add_node") (cgraph @-> tensor @-> returning void)

  (** [graph_overhead ()] returns the memory overhead of a default-sized graph structure.
      - returns Overhead in bytes. *)
  let graph_overhead = foreign (ns "graph_overhead") (void @-> returning size_t)

  (** [graph_overhead_custom size grads] returns the memory overhead for a custom-sized graph.
      - [size] The maximum number of nodes.
      - [grads] Whether the graph stores gradients.
      - returns Overhead in bytes. *)
  let graph_overhead_custom = foreign (ns "graph_overhead_custom") (size_t @-> bool @-> returning size_t)

  (** [graph_get_tensor graph name] retrieves a tensor from the graph by its name.
      - [graph] The computation graph.
      - [name] The name of the tensor.
      - returns The tensor, or NULL if not found. *)
  let graph_get_tensor = foreign (ns "graph_get_tensor") (cgraph @-> string @-> returning tensor)

  (** [graph_get_grad graph tensor] retrieves the gradient tensor associated with a given tensor in the graph.
      - [graph] The computation graph.
      - [tensor] The tensor whose gradient is requested. c
      - returns The gradient tensor, or NULL if gradients are not stored or not computed. *)
  let graph_get_grad = foreign (ns "graph_get_grad") (cgraph @-> tensor @-> returning tensor)

  (** [graph_get_grad_acc graph tensor] retrieves the accumulated gradient tensor. (Likely internal use).
      - [graph] The computation graph.
      - [tensor] The tensor.
      - returns The accumulated gradient tensor. *)
  let graph_get_grad_acc = foreign (ns "graph_get_grad_acc") (cgraph @-> tensor @-> returning tensor)

  (** [graph_print graph] prints information about the computation graph to stderr.
      - [graph] The computation graph. *)
  let graph_print = foreign (ns "graph_print") (cgraph @-> returning void)

  (** [graph_dump_dot gf gb filename] dumps the computation graph(s) in DOT format to a file.
      - [gf] Forward graph (optional).
      - [gb] Backward graph (optional).
      - [filename] The output filename. *)
  let graph_dump_dot = foreign (ns "graph_dump_dot") (cgraph @-> cgraph @-> string @-> returning void)
end
