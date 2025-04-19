open Ctypes
module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types

  (* Context *)

  (** [init params] initializes a ggml context.
      @param params Initialization parameters.
      @return A new ggml context. *)
  let init = foreign (ns "init") (InitParams.t @-> returning context)

  (** [free ctx] frees the memory associated with a ggml context.
      @param ctx The context to free. *)
  let free = foreign (ns "free") (context @-> returning void)

  (** [used_mem ctx] returns the amount of memory used by the context in bytes.
      @param ctx The context.
      @return Memory used in bytes. *)
  let used_mem = foreign (ns "used_mem") (context @-> returning size_t)

  (** [reset ctx] resets the context, clearing its internal state but keeping allocated memory.
      @param ctx The context to reset. *)
  let reset = foreign (ns "reset") (context @-> returning void)

  (** [guid_matches guid_a guid_b] checks if two GUIDs are equal.
      @param guid_a First GUID.
      @param guid_b Second GUID.
      @return True if the GUIDs match, false otherwise. *)
  let guid_matches = foreign (ns "guid_matches") (guid_t @-> guid_t @-> returning bool)

  (* Time Functions *)

  (** [time_init ()] initializes the internal timer. Call once at program start. *)
  let time_init = foreign (ns "time_init") (void @-> returning void)

  (** [time_ms ()] returns the current time in milliseconds.
      @return Time in milliseconds. *)
  let time_ms = foreign (ns "time_ms") (void @-> returning int64_t)

  (** [time_us ()] returns the current time in microseconds.
      @return Time in microseconds. *)
  let time_us = foreign (ns "time_us") (void @-> returning int64_t)

  (** [cycles ()] returns the current CPU cycle count.
      @return CPU cycle count. *)
  let cycles = foreign (ns "cycles") (void @-> returning int64_t)

  (** [cycles_per_ms ()] returns the number of CPU cycles per millisecond.
      @return CPU cycles per millisecond. *)
  let cycles_per_ms = foreign (ns "cycles_per_ms") (void @-> returning int64_t)

  (* File Handling *)

  (** [fopen fname mode] opens a file, accepting UTF-8 paths even on Windows.
      @param fname The filename (UTF-8).
      @param mode The file opening mode (e.g., "rb", "wb").
      @return A file pointer (represented as `ptr void`), or NULL on failure. *)
  let fopen = foreign (ns "fopen") (string @-> string @-> returning (ptr void))

  (* Printing *)

  (** [print_object obj] prints information about a ggml object to stderr.
      @param obj The object to print. *)
  let print_object = foreign (ns "print_object") (object' @-> returning void)

  (** [print_objects ctx] prints information about all objects in a context to stderr.
      @param ctx The context. *)
  let print_objects = foreign (ns "print_objects") (context @-> returning void)

  (* Types / Ops Info *)

  (** [type_name typ] returns the name of the ggml type.
      @param typ The ggml type.
      @return The name of the type. *)
  let type_name = foreign (ns "type_name") (typ @-> returning string)

  (** [op_name op] returns the name of the ggml operation.
      @param op The ggml operation.
      @return The name of the operation. *)
  let op_name = foreign (ns "op_name") (op @-> returning string)

  (** [op_symbol op] returns the symbolic representation of the ggml operation (e.g., "+", "*").
      @param op The ggml operation.
      @return The symbol of the operation. *)
  let op_symbol = foreign (ns "op_symbol") (op @-> returning string)

  (** [unary_op_name op] returns the name of the ggml unary operation.
      @param op The ggml unary operation.
      @return The name of the unary operation. *)
  let unary_op_name = foreign (ns "unary_op_name") (unary_op @-> returning string)

  (** [op_desc tensor] returns a description of the operation that produced the tensor.
      @param tensor The tensor.
      @return A description string (unary op name or op name). *)
  let op_desc = foreign (ns "op_desc") (tensor @-> returning string)

  (** [blck_size typ] returns the block size for a given ggml type.
      @param typ The ggml type.
      @return The block size (number of elements in a block). *)
  let blck_size = foreign (ns "blck_size") (typ @-> returning int64_t)

  (** [type_size typ] returns the size in bytes for all elements in a block of the given type.
      @param typ The ggml type.
      @return The size in bytes of a block. *)
  let type_size = foreign (ns "type_size") (typ @-> returning size_t)

  (** [row_size typ ne] returns the size in bytes for a row containing `ne` elements of the given type.
      @param typ The ggml type.
      @param ne The number of elements in the row.
      @return The size in bytes of the row. *)
  let row_size = foreign (ns "row_size") (typ @-> int64_t @-> returning size_t)

  (** [is_quantized typ] checks if the ggml type is a quantized type.
      @param typ The ggml type.
      @return True if the type is quantized, false otherwise. *)
  let is_quantized = foreign (ns "is_quantized") (typ @-> returning bool)

  (** [ftype_to_ggml_type ftype] converts a file type enum to a ggml type enum.
      @param ftype The file type enum value.
      @return The corresponding ggml type enum value. *)
  let ftype_to_ggml_type = foreign (ns "ftype_to_ggml_type") (ftype @-> returning typ)

  (* Tensor Info *)

  (** [element_size tensor] returns the size in bytes of a single element in the tensor.
      @param tensor The tensor.
      @return The element size in bytes. *)
  let element_size = foreign (ns "element_size") (tensor @-> returning size_t)

  (** [nelements tensor] returns the total number of elements in the tensor.
      @param tensor The tensor.
      @return The number of elements. *)
  let nelements = foreign (ns "nelements") (tensor @-> returning int64_t)

  (** [nbytes tensor] returns the total size in bytes of the tensor's data.
      @param tensor The tensor.
      @return The size in bytes. *)
  let nbytes = foreign (ns "nbytes") (tensor @-> returning size_t)

  (** [nbytes_pad tensor] returns the padded size in bytes of the tensor's data (aligned to GGML_MEM_ALIGN).
      @param tensor The tensor.
      @return The padded size in bytes. *)
  let nbytes_pad = foreign (ns "nbytes_pad") (tensor @-> returning size_t)

  (** [nrows tensor] returns the number of rows in the tensor (product of dimensions >= 1).
      @param tensor The tensor.
      @return The number of rows. *)
  let nrows = foreign (ns "nrows") (tensor @-> returning int64_t)

  (** [is_transposed tensor] checks if the tensor is transposed (swapped strides for dims 0 and 1).
      @param tensor The tensor.
      @return True if transposed, false otherwise. *)
  let is_transposed = foreign (ns "is_transposed") (tensor @-> returning bool)

  (** [is_permuted tensor] checks if the tensor is permuted (strides differ from canonical order).
      @param tensor The tensor.
      @return True if permuted, false otherwise. *)
  let is_permuted = foreign (ns "is_permuted") (tensor @-> returning bool)

  (** [is_empty tensor] checks if the tensor has zero elements.
      @param tensor The tensor.
      @return True if empty, false otherwise. *)
  let is_empty = foreign (ns "is_empty") (tensor @-> returning bool)

  (** [is_scalar tensor] checks if the tensor is a scalar (all dimensions are 1).
      @param tensor The tensor.
      @return True if scalar, false otherwise. *)
  let is_scalar = foreign (ns "is_scalar") (tensor @-> returning bool)

  (** [is_vector tensor] checks if the tensor is a vector (one non-unity dimension).
      @param tensor The tensor.
      @return True if vector, false otherwise. *)
  let is_vector = foreign (ns "is_vector") (tensor @-> returning bool)

  (** [is_matrix tensor] checks if the tensor is a matrix (two non-unity dimensions).
      @param tensor The tensor.
      @return True if matrix, false otherwise. *)
  let is_matrix = foreign (ns "is_matrix") (tensor @-> returning bool)

  (** [is_3d tensor] checks if the tensor has exactly three non-unity dimensions.
      @param tensor The tensor.
      @return True if 3D, false otherwise. *)
  let is_3d = foreign (ns "is_3d") (tensor @-> returning bool)

  (** [n_dims tensor] returns the number of dimensions of the tensor (returns 1 for scalars).
      @param tensor The tensor.
      @return The number of dimensions. *)
  let n_dims = foreign (ns "n_dims") (tensor @-> returning int)

  (** [is_contiguous tensor] checks if the tensor's data is laid out contiguously in memory.
      @param tensor The tensor.
      @return True if contiguous, false otherwise. *)
  let is_contiguous = foreign (ns "is_contiguous") (tensor @-> returning bool)

  (** [is_contiguous_0 tensor] alias for `is_contiguous`.
      @param tensor The tensor.
      @return True if contiguous, false otherwise. *)
  let is_contiguous_0 = foreign (ns "is_contiguous_0") (tensor @-> returning bool)

  (** [is_contiguous_1 tensor] checks if the tensor is contiguous for dimensions >= 1.
      @param tensor The tensor.
      @return True if contiguous for dims >= 1, false otherwise. *)
  let is_contiguous_1 = foreign (ns "is_contiguous_1") (tensor @-> returning bool)

  (** [is_contiguous_2 tensor] checks if the tensor is contiguous for dimensions >= 2.
      @param tensor The tensor.
      @return True if contiguous for dims >= 2, false otherwise. *)
  let is_contiguous_2 = foreign (ns "is_contiguous_2") (tensor @-> returning bool)

  (** [are_same_shape t0 t1] checks if two tensors have the same shape.
      @param t0 First tensor.
      @param t1 Second tensor.
      @return True if shapes are identical, false otherwise. *)
  let are_same_shape = foreign (ns "are_same_shape") (tensor @-> tensor @-> returning bool)

  (** [are_same_stride t0 t1] checks if two tensors have the same strides.
      @param t0 First tensor.
      @param t1 Second tensor.
      @return True if strides are identical, false otherwise. *)
  let are_same_stride = foreign (ns "are_same_stride") (tensor @-> tensor @-> returning bool)

  (** [can_repeat t0 t1] checks if tensor `t0` can be repeated (broadcasted) to match the shape of `t1`.
      @param t0 The tensor to potentially repeat.
      @param t1 The target shape tensor.
      @return True if `t0` can be repeated to match `t1`, false otherwise. *)
  let can_repeat = foreign (ns "can_repeat") (tensor @-> tensor @-> returning bool)

  (** [tensor_overhead ()] returns the memory overhead of the ggml_tensor struct itself.
      @return Overhead in bytes. *)
  let tensor_overhead = foreign (ns "tensor_overhead") (void @-> returning size_t)

  (** [validate_row_data typ data nbytes] validates if the data buffer is suitable for the given type and size.
      @param typ The ggml type.
      @param data Pointer to the data buffer.
      @param nbytes Size of the data buffer in bytes.
      @return True if the data is valid, false otherwise. *)
  let validate_row_data = foreign (ns "validate_row_data") (typ @-> ptr void @-> size_t @-> returning bool)

  (* Tensor Creation *)

  (** [new_tensor ctx typ n_dims ne] creates a new tensor with the specified type and dimensions.
      @param ctx The context.
      @param typ The tensor type.
      @param n_dims The number of dimensions.
      @param ne Pointer to an array containing the size of each dimension.
      @return The new tensor. *)
  let new_tensor = foreign (ns "new_tensor") (context @-> typ @-> int @-> ptr int64_t @-> returning tensor)

  (** [new_tensor_1d ctx typ ne0] creates a new 1D tensor.
      @param ctx The context.
      @param typ The tensor type.
      @param ne0 Size of the first dimension.
      @return The new tensor. *)
  let new_tensor_1d = foreign (ns "new_tensor_1d") (context @-> typ @-> int64_t @-> returning tensor)

  (** [new_tensor_2d ctx typ ne0 ne1] creates a new 2D tensor.
      @param ctx The context.
      @param typ The tensor type.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @return The new tensor. *)
  let new_tensor_2d = foreign (ns "new_tensor_2d") (context @-> typ @-> int64_t @-> int64_t @-> returning tensor)

  (** [new_tensor_3d ctx typ ne0 ne1 ne2] creates a new 3D tensor.
      @param ctx The context.
      @param typ The tensor type.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @param ne2 Size of the third dimension.
      @return The new tensor. *)
  let new_tensor_3d =
    foreign (ns "new_tensor_3d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [new_tensor_4d ctx typ ne0 ne1 ne2 ne3] creates a new 4D tensor.
      @param ctx The context.
      @param typ The tensor type.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @param ne2 Size of the third dimension.
      @param ne3 Size of the fourth dimension.
      @return The new tensor. *)
  let new_tensor_4d =
    foreign (ns "new_tensor_4d") (context @-> typ @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (* Buffer Creation *)

  (** [new_buffer ctx nbytes] allocates a buffer of the specified size within the context.
      @param ctx The context.
      @param nbytes The size of the buffer in bytes.
      @return A pointer to the allocated buffer. *)
  let new_buffer = foreign (ns "new_buffer") (context @-> size_t @-> returning (ptr void))

  (* Tensor Duplication / Viewing *)

  (** [dup_tensor ctx src] duplicates a tensor, including its data.
      @param ctx The context.
      @param src The source tensor to duplicate.
      @return The duplicated tensor. *)
  let dup_tensor = foreign (ns "dup_tensor") (context @-> tensor @-> returning tensor)

  (** [view_tensor ctx src] creates a view of a tensor. The view shares data with the original tensor.
      @param ctx The context.
      @param src The source tensor to view.
      @return The view tensor. *)
  let view_tensor = foreign (ns "view_tensor") (context @-> tensor @-> returning tensor)

  (* Tensor Duplication / Viewing *)

  (** [aet_first_tensor ctx] returns the first tensor allocated in the context.
      @param ctx The context.
      @return The first tensor, or NULL if none exist. *)
  let get_first_tensor = foreign (ns "get_first_tensor") (context @-> returning tensor)

  (** [get_next_tensor ctx tensor] returns the next tensor allocated in the context after the given tensor.
      @param ctx The context.
      @param tensor The current tensor.
      @return The next tensor, or NULL if it's the last one. *)
  let get_next_tensor = foreign (ns "get_next_tensor") (context @-> tensor @-> returning tensor)

  (** [get_tensor ctx name] retrieves a tensor from the context by its name.
      @param ctx The context.
      @param name The name of the tensor.
      @return The tensor with the specified name, or NULL if not found. *)
  let get_tensor = foreign (ns "get_tensor") (context @-> string @-> returning tensor)

  (* Indexing *)

  (** [unravel_index tensor i i0 i1 i2 i3] converts a flat index `i` into multi-dimensional coordinates for the tensor.
      @param tensor The tensor.
      @param i The flat index.
      @param i0 Pointer to store the coordinate for the first dimension.
      @param i1 Pointer to store the coordinate for the second dimension.
      @param i2 Pointer to store the coordinate for the third dimension.
      @param i3 Pointer to store the coordinate for the fourth dimension. *)
  let unravel_index =
    foreign (ns "unravel_index")
      (tensor @-> int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> ptr int64_t @-> returning void)

  (* Op Info *)

  (** [get_unary_op tensor] returns the unary operation associated with the tensor, if any.
      @param tensor The tensor.
      @return The unary operation enum value. *)
  let get_unary_op = foreign (ns "get_unary_op") (tensor @-> returning unary_op)

  (* Data Access *)

  (** [get_data tensor] returns a raw pointer to the tensor's data.
      @param tensor The tensor.
      @return A void pointer to the data. *)
  let get_data = foreign (ns "get_data") (tensor @-> returning (ptr void))

  (** [get_data_f32 tensor] returns a pointer to the tensor's data, cast to float*.
      @param tensor The tensor.
      @return A float pointer to the data. *)
  let get_data_f32 = foreign (ns "get_data_f32") (tensor @-> returning (ptr float))

  (* Tensor Naming *)

  (** [get_name tensor] returns the name of the tensor.
      @param tensor The tensor.
      @return The name as a string. *)
  let get_name = foreign (ns "get_name") (tensor @-> returning string)

  (** [set_name tensor name] sets the name of the tensor.
      @param tensor The tensor to name.
      @param name The desired name.
      @return The tensor itself. *)
  let set_name = foreign (ns "set_name") (tensor @-> string @-> returning tensor)
  (* ggml_format_name is variadic, skipping *)

  (* Tensor Flags *)

  (** [set_input tensor] marks the tensor as an input for the compute graph.
      @param tensor The tensor to mark. *)
  let set_input = foreign (ns "set_input") (tensor @-> returning void)

  (** [set_output tensor] marks the tensor as an output for the compute graph.
      @param tensor The tensor to mark. *)
  let set_output = foreign (ns "set_output") (tensor @-> returning void)

  (** [set_param ctx tensor] marks the tensor as containing trainable parameters.
      @param ctx The context.
      @param tensor The tensor to mark. *)
  let set_param = foreign (ns "set_param") (context @-> tensor @-> returning void)

  (** [set_loss tensor] marks the tensor as defining loss for numerical optimization. Multiple loss tensors add up.
      @param tensor The tensor to mark. *)
  let set_loss = foreign (ns "set_loss") (tensor @-> returning void)

  (* Operations with backpropagation *)

  (** [dup ctx a] duplicates tensor `a`.
      @param ctx The context.
      @param a The tensor to duplicate.
      @return The duplicated tensor. *)
  let dup = foreign (ns "dup") (context @-> tensor @-> returning tensor)

  (** [dup_inplace ctx a] duplicates tensor `a` in-place. Returns a view of `a`.
      @param ctx The context.
      @param a The tensor to duplicate.
      @return A view of the tensor `a`. *)
  let dup_inplace = foreign (ns "dup_inplace") (context @-> tensor @-> returning tensor)

  (** [add ctx a b] computes `a + b`.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @return The resulting tensor. *)
  let add = foreign (ns "add") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add_inplace ctx a b] computes `a + b` in-place, modifying `a`.
      @param ctx The context.
      @param a First tensor (modified).
      @param b Second tensor.
      @return The modified tensor `a`. *)
  let add_inplace = foreign (ns "add_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add_cast ctx a b typ] computes `a + b` and casts the result to `typ`.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @param typ The target type for the result.
      @return The resulting tensor. *)
  let add_cast = foreign (ns "add_cast") (context @-> tensor @-> tensor @-> typ @-> returning tensor)

  (** [add1 ctx a b] computes `a + b*1`. Adds the scalar `b` to each element of `a`.
      @param ctx The context.
      @param a The tensor.
      @param b The scalar tensor to add.
      @return The resulting tensor. *)
  let add1 = foreign (ns "add1") (context @-> tensor @-> tensor @-> returning tensor)

  (** [add1_inplace ctx a b] computes `a + b*1` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @param b The scalar tensor to add.
      @return The modified tensor `a`. *)
  let add1_inplace = foreign (ns "add1_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [acc ctx a b nb1 nb2 nb3 offset] accumulates `b` into a view of `a`. `dst = a; view(dst, nb1, nb2, nb3, offset) +=
      b`.
      @param ctx The context.
      @param a The destination tensor.
      @param b The tensor to accumulate.
      @param nb1 Stride for the first dimension of the view.
      @param nb2 Stride for the second dimension of the view.
      @param nb3 Stride for the third dimension of the view.
      @param offset Offset in bytes for the view.
      @return The modified tensor `a`. *)
  let acc =
    foreign (ns "acc") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [acc_inplace ctx a b nb1 nb2 nb3 offset] accumulates `b` into a view of `a` in-place.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The tensor to accumulate.
      @param nb1 Stride for the first dimension of the view.
      @param nb2 Stride for the second dimension of the view.
      @param nb3 Stride for the third dimension of the view.
      @param offset Offset in bytes for the view.
      @return The modified tensor `a`. *)
  let acc_inplace =
    foreign (ns "acc_inplace")
      (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [sub ctx a b] computes `a - b`.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @return The resulting tensor. *)
  let sub = foreign (ns "sub") (context @-> tensor @-> tensor @-> returning tensor)

  (** [sub_inplace ctx a b] computes `a - b` in-place, modifying `a`.
      @param ctx The context.
      @param a First tensor (modified).
      @param b Second tensor.
      @return The modified tensor `a`. *)
  let sub_inplace = foreign (ns "sub_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul ctx a b] computes element-wise multiplication `a * b`.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @return The resulting tensor. *)
  let mul = foreign (ns "mul") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul_inplace ctx a b] computes element-wise `a * b` in-place, modifying `a`.
      @param ctx The context.
      @param a First tensor (modified).
      @param b Second tensor.
      @return The modified tensor `a`. *)
  let mul_inplace = foreign (ns "mul_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [div ctx a b] computes element-wise division `a / b`.
      @param ctx The context.
      @param a Numerator tensor.
      @param b Denominator tensor.
      @return The resulting tensor. *)
  let div = foreign (ns "div") (context @-> tensor @-> tensor @-> returning tensor)

  (** [div_inplace ctx a b] computes element-wise `a / b` in-place, modifying `a`.
      @param ctx The context.
      @param a Numerator tensor (modified).
      @param b Denominator tensor.
      @return The modified tensor `a`. *)
  let div_inplace = foreign (ns "div_inplace") (context @-> tensor @-> tensor @-> returning tensor)

  (** [sqr ctx a] computes element-wise square `a^2`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let sqr = foreign (ns "sqr") (context @-> tensor @-> returning tensor)

  (** [sqr_inplace ctx a] computes element-wise `a^2` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let sqr_inplace = foreign (ns "sqr_inplace") (context @-> tensor @-> returning tensor)

  (** [sqrt ctx a] computes element-wise square root `sqrt(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let sqrt = foreign (ns "sqrt") (context @-> tensor @-> returning tensor)

  (** [sqrt_inplace ctx a] computes element-wise `sqrt(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let sqrt_inplace = foreign (ns "sqrt_inplace") (context @-> tensor @-> returning tensor)

  (** [log ctx a] computes element-wise natural logarithm `log(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let log = foreign (ns "log") (context @-> tensor @-> returning tensor)

  (** [log_inplace ctx a] computes element-wise `log(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let log_inplace = foreign (ns "log_inplace") (context @-> tensor @-> returning tensor)

  (** [sin ctx a] computes element-wise sine `sin(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let sin = foreign (ns "sin") (context @-> tensor @-> returning tensor)

  (** [sin_inplace ctx a] computes element-wise `sin(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let sin_inplace = foreign (ns "sin_inplace") (context @-> tensor @-> returning tensor)

  (** [cos ctx a] computes element-wise cosine `cos(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let cos = foreign (ns "cos") (context @-> tensor @-> returning tensor)

  (** [cos_inplace ctx a] computes element-wise `cos(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let cos_inplace = foreign (ns "cos_inplace") (context @-> tensor @-> returning tensor)

  (** [sum ctx a] computes the sum of all elements in `a`. Returns a scalar tensor.
      @param ctx The context.
      @param a The tensor.
      @return A scalar tensor containing the sum. *)
  let sum = foreign (ns "sum") (context @-> tensor @-> returning tensor)

  (** [sum_rows ctx a] computes the sum along the first dimension (rows). Input shape [a,b,c,d] -> output shape
      [1,b,c,d].
      @param ctx The context.
      @param a The tensor.
      @return A tensor containing the row sums. *)
  let sum_rows = foreign (ns "sum_rows") (context @-> tensor @-> returning tensor)

  (** [mean ctx a] computes the mean along the first dimension (rows).
      @param ctx The context.
      @param a The tensor.
      @return A tensor containing the row means. *)
  let mean = foreign (ns "mean") (context @-> tensor @-> returning tensor)

  (** [argmax ctx a] computes the index of the maximum value along the first dimension (rows).
      @param ctx The context.
      @param a The tensor.
      @return A tensor containing the indices of the maximum values. *)
  let argmax = foreign (ns "argmax") (context @-> tensor @-> returning tensor)

  (** [count_equal ctx a b] counts the number of equal elements between tensors `a` and `b`. Returns a scalar tensor.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @return A scalar tensor containing the count. *)
  let count_equal = foreign (ns "count_equal") (context @-> tensor @-> tensor @-> returning tensor)

  (** [repeat ctx a b] repeats tensor `a` to match the shape of tensor `b`. If `a` already has the same shape as `b` and
      is not a parameter, `a` is returned directly.
      @param ctx The context.
      @param a The tensor to repeat.
      @param b The tensor defining the target shape.
      @return The repeated tensor. *)
  let repeat = foreign (ns "repeat") (context @-> tensor @-> tensor @-> returning tensor)

  (** [repeat_back ctx a b] sums repetitions in `a` back into the shape of `b`. This is the backward operation for
      `repeat`.
      @param ctx The context.
      @param a The tensor containing repetitions (gradient of `repeat` output).
      @param b The tensor defining the target shape (original input to `repeat`).
      @return The resulting tensor with summed repetitions. *)
  let repeat_back = foreign (ns "repeat_back") (context @-> tensor @-> tensor @-> returning tensor)

  (** [concat ctx a b dim] concatenates tensors `a` and `b` along the specified dimension `dim`.
      @param ctx The context.
      @param a First tensor.
      @param b Second tensor.
      @param dim The dimension along which to concatenate.
      @return The concatenated tensor. *)
  let concat = foreign (ns "concat") (context @-> tensor @-> tensor @-> int @-> returning tensor)

  (** [abs ctx a] computes element-wise absolute value `abs(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let abs = foreign (ns "abs") (context @-> tensor @-> returning tensor)

  (** [abs_inplace ctx a] computes element-wise `abs(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let abs_inplace = foreign (ns "abs_inplace") (context @-> tensor @-> returning tensor)

  (** [sgn ctx a] computes element-wise sign `sgn(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let sgn = foreign (ns "sgn") (context @-> tensor @-> returning tensor)

  (** [sgn_inplace ctx a] computes element-wise `sgn(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let sgn_inplace = foreign (ns "sgn_inplace") (context @-> tensor @-> returning tensor)

  (** [neg ctx a] computes element-wise negation `-a`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let neg = foreign (ns "neg") (context @-> tensor @-> returning tensor)

  (** [neg_inplace ctx a] computes element-wise `-a` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let neg_inplace = foreign (ns "neg_inplace") (context @-> tensor @-> returning tensor)

  (** [step ctx a] computes element-wise step function (1 if x > 0, 0 otherwise).
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let step = foreign (ns "step") (context @-> tensor @-> returning tensor)

  (** [step_inplace ctx a] computes element-wise step function in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let step_inplace = foreign (ns "step_inplace") (context @-> tensor @-> returning tensor)

  (** [tanh ctx a] computes element-wise hyperbolic tangent `tanh(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let tanh = foreign (ns "tanh") (context @-> tensor @-> returning tensor)

  (** [tanh_inplace ctx a] computes element-wise `tanh(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let tanh_inplace = foreign (ns "tanh_inplace") (context @-> tensor @-> returning tensor)

  (** [elu ctx a] computes element-wise Exponential Linear Unit `elu(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let elu = foreign (ns "elu") (context @-> tensor @-> returning tensor)

  (** [elu_inplace ctx a] computes element-wise `elu(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let elu_inplace = foreign (ns "elu_inplace") (context @-> tensor @-> returning tensor)

  (** [relu ctx a] computes element-wise Rectified Linear Unit `relu(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let relu = foreign (ns "relu") (context @-> tensor @-> returning tensor)

  (** [leaky_relu ctx a negative_slope inplace] computes element-wise Leaky Rectified Linear Unit.
      @param ctx The context.
      @param a The tensor.
      @param negative_slope The slope for negative values.
      @param inplace Whether to perform the operation in-place.
      @return The resulting tensor (or modified `a` if `inplace` is true). *)
  let leaky_relu = foreign (ns "leaky_relu") (context @-> tensor @-> float @-> bool @-> returning tensor)

  (** [relu_inplace ctx a] computes element-wise `relu(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let relu_inplace = foreign (ns "relu_inplace") (context @-> tensor @-> returning tensor)

  (** [sigmoid ctx a] computes element-wise sigmoid function `sigmoid(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let sigmoid = foreign (ns "sigmoid") (context @-> tensor @-> returning tensor)

  (** [sigmoid_inplace ctx a] computes element-wise `sigmoid(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let sigmoid_inplace = foreign (ns "sigmoid_inplace") (context @-> tensor @-> returning tensor)

  (** [gelu ctx a] computes element-wise Gaussian Error Linear Unit `gelu(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let gelu = foreign (ns "gelu") (context @-> tensor @-> returning tensor)

  (** [gelu_inplace ctx a] computes element-wise `gelu(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let gelu_inplace = foreign (ns "gelu_inplace") (context @-> tensor @-> returning tensor)

  (** [gelu_quick ctx a] computes element-wise approximate GELU `gelu_quick(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let gelu_quick = foreign (ns "gelu_quick") (context @-> tensor @-> returning tensor)

  (** [gelu_quick_inplace ctx a] computes element-wise `gelu_quick(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let gelu_quick_inplace = foreign (ns "gelu_quick_inplace") (context @-> tensor @-> returning tensor)

  (** [silu ctx a] computes element-wise Sigmoid Linear Unit `silu(a) = a * sigmoid(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let silu = foreign (ns "silu") (context @-> tensor @-> returning tensor)

  (** [silu_inplace ctx a] computes element-wise `silu(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let silu_inplace = foreign (ns "silu_inplace") (context @-> tensor @-> returning tensor)

  (** [silu_back ctx a b] computes the backward pass for SiLU.
      @param ctx The context.
      @param a The input tensor `x` from the forward pass.
      @param b The gradient `dy` from the output.
      @return The gradient `dx`. *)
  let silu_back = foreign (ns "silu_back") (context @-> tensor @-> tensor @-> returning tensor)

  (** [hardswish ctx a] computes element-wise Hardswish `hardswish(a) = a * relu6(a + 3) / 6`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let hardswish = foreign (ns "hardswish") (context @-> tensor @-> returning tensor)

  (** [hardsigmoid ctx a] computes element-wise Hardsigmoid `hardsigmoid(a) = relu6(a + 3) / 6`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let hardsigmoid = foreign (ns "hardsigmoid") (context @-> tensor @-> returning tensor)

  (** [exp ctx a] computes element-wise exponentiation `exp(a)`.
      @param ctx The context.
      @param a The tensor.
      @return The resulting tensor. *)
  let exp = foreign (ns "exp") (context @-> tensor @-> returning tensor)

  (** [exp_inplace ctx a] computes element-wise `exp(a)` in-place, modifying `a`.
      @param ctx The context.
      @param a The tensor (modified).
      @return The modified tensor `a`. *)
  let exp_inplace = foreign (ns "exp_inplace") (context @-> tensor @-> returning tensor)

  (** [norm ctx a eps] normalizes `a` along the first dimension (rows).
      @param ctx The context.
      @param a The tensor to normalize.
      @param eps Epsilon value for numerical stability.
      @return The normalized tensor. *)
  let norm = foreign (ns "norm") (context @-> tensor @-> float @-> returning tensor)

  (** [norm_inplace ctx a eps] normalizes `a` along rows in-place.
      @param ctx The context.
      @param a The tensor to normalize (modified).
      @param eps Epsilon value for numerical stability.
      @return The modified tensor `a`. *)
  let norm_inplace = foreign (ns "norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm ctx a eps] computes Root Mean Square normalization along rows.
      @param ctx The context.
      @param a The tensor to normalize.
      @param eps Epsilon value for numerical stability.
      @return The normalized tensor. *)
  let rms_norm = foreign (ns "rms_norm") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm_inplace ctx a eps] computes RMS normalization along rows in-place.
      @param ctx The context.
      @param a The tensor to normalize (modified).
      @param eps Epsilon value for numerical stability.
      @return The modified tensor `a`. *)
  let rms_norm_inplace = foreign (ns "rms_norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [group_norm ctx a n_groups eps] computes Group Normalization. Normalizes along ne0*ne1*n_groups.
      @param ctx The context.
      @param a The tensor to normalize.
      @param n_groups The number of groups.
      @param eps Epsilon value for numerical stability.
      @return The normalized tensor. *)
  let group_norm = foreign (ns "group_norm") (context @-> tensor @-> int @-> float @-> returning tensor)

  (** [group_norm_inplace ctx a n_groups eps] computes Group Normalization in-place.
      @param ctx The context.
      @param a The tensor to normalize (modified).
      @param n_groups The number of groups.
      @param eps Epsilon value for numerical stability.
      @return The modified tensor `a`. *)
  let group_norm_inplace = foreign (ns "group_norm_inplace") (context @-> tensor @-> int @-> float @-> returning tensor)

  (** [l2_norm ctx a eps] computes L2 normalization along rows.
      @param ctx The context.
      @param a The tensor to normalize.
      @param eps Epsilon value for numerical stability.
      @return The normalized tensor. *)
  let l2_norm = foreign (ns "l2_norm") (context @-> tensor @-> float @-> returning tensor)

  (** [l2_norm_inplace ctx a eps] computes L2 normalization along rows in-place.
      @param ctx The context.
      @param a The tensor to normalize (modified).
      @param eps Epsilon value for numerical stability.
      @return The modified tensor `a`. *)
  let l2_norm_inplace = foreign (ns "l2_norm_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [rms_norm_back ctx a b eps] computes the backward pass for RMS normalization.
      @param ctx The context.
      @param a The input tensor `x` from the forward pass.
      @param b The gradient `dy` from the output.
      @param eps Epsilon value used in the forward pass.
      @return The gradient `dx`. *)
  let rms_norm_back = foreign (ns "rms_norm_back") (context @-> tensor @-> tensor @-> float @-> returning tensor)

  (** [mul_mat ctx a b] computes matrix multiplication `a * b^T`. A: [..., n, k], B: [..., m, k] -> Result: [..., m, n].
      @param ctx The context.
      @param a First matrix.
      @param b Second matrix (transposed internally).
      @return The resulting matrix. *)
  let mul_mat = foreign (ns "mul_mat") (context @-> tensor @-> tensor @-> returning tensor)

  (** [mul_mat_set_prec a prec] changes the precision used for the matrix multiplication involving tensor `a`.
      @param a The tensor involved in the `mul_mat` operation (typically the output tensor).
      @param prec The desired precision (e.g., `GGML_PREC_F32`). *)
  let mul_mat_set_prec = foreign (ns "mul_mat_set_prec") (tensor @-> prec @-> returning void)

  (** [mul_mat_id ctx as b ids] performs indirect matrix multiplication using IDs.
      @param ctx The context.
      @param as Tensor containing multiple matrices.
      @param b The second matrix.
      @param ids Tensor containing indices to select matrices from `as`.
      @return The resulting matrix. *)
  let mul_mat_id = foreign (ns "mul_mat_id") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [out_prod ctx a b] computes the outer product of vectors `a` and `b`. A: [n, ...], B: [m, ...] -> Result:
      [m, n, ...].
      @param ctx The context.
      @param a First vector.
      @param b Second vector.
      @return The resulting matrix (outer product). *)
  let out_prod = foreign (ns "out_prod") (context @-> tensor @-> tensor @-> returning tensor)

  (** [scale ctx a s] scales tensor `a` by scalar `s`.
      @param ctx The context.
      @param a The tensor to scale.
      @param s The scaling factor.
      @return The scaled tensor. *)
  let scale = foreign (ns "scale") (context @-> tensor @-> float @-> returning tensor)

  (** [scale_inplace ctx a s] scales tensor `a` by scalar `s` in-place. Returns a view of `a`.
      @param ctx The context.
      @param a The tensor to scale (modified).
      @param s The scaling factor.
      @return A view of the modified tensor `a`. *)
  let scale_inplace = foreign (ns "scale_inplace") (context @-> tensor @-> float @-> returning tensor)

  (** [set ctx a b nb1 nb2 nb3 offset] sets the elements of a view of `a` to the values of `b`. Returns the modified
      `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The source tensor.
      @param nb1 Stride for the first dimension of the view.
      @param nb2 Stride for the second dimension of the view.
      @param nb3 Stride for the third dimension of the view.
      @param offset Offset in bytes for the view.
      @return The modified tensor `a`. *)
  let set =
    foreign (ns "set") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [set_inplace ctx a b nb1 nb2 nb3 offset] sets the elements of a view of `a` to the values of `b`. Returns a view
      of `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The source tensor.
      @param nb1 Stride for the first dimension of the view.
      @param nb2 Stride for the second dimension of the view.
      @param nb3 Stride for the third dimension of the view.
      @param offset Offset in bytes for the view.
      @return A view of the modified tensor `a`. *)
  let set_inplace =
    foreign (ns "set_inplace")
      (context @-> tensor @-> tensor @-> size_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [set_1d ctx a b offset] sets elements of `a` starting at `offset` to the values of 1D tensor `b`. Returns modified
      `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The 1D source tensor.
      @param offset Offset in bytes.
      @return The modified tensor `a`. *)
  let set_1d = foreign (ns "set_1d") (context @-> tensor @-> tensor @-> size_t @-> returning tensor)

  (** [set_1d_inplace ctx a b offset] sets elements of `a` starting at `offset` to the values of 1D tensor `b`. Returns
      a view of `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The 1D source tensor.
      @param offset Offset in bytes.
      @return A view of the modified tensor `a`. *)
  let set_1d_inplace = foreign (ns "set_1d_inplace") (context @-> tensor @-> tensor @-> size_t @-> returning tensor)

  (** [set_2d ctx a b nb1 offset] sets elements of a 2D view of `a` to the values of `b`. Returns modified `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The source tensor.
      @param nb1 Stride for the first dimension of the view.
      @param offset Offset in bytes.
      @return The modified tensor `a`. *)
  let set_2d = foreign (ns "set_2d") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> returning tensor)

  (** [set_2d_inplace ctx a b nb1 offset] sets elements of a 2D view of `a` to the values of `b`. Returns a view of `a`.
      @param ctx The context.
      @param a The destination tensor (modified).
      @param b The source tensor.
      @param nb1 Stride for the first dimension of the view.
      @param offset Offset in bytes.
      @return A view of the modified tensor `a`. *)
  let set_2d_inplace =
    foreign (ns "set_2d_inplace") (context @-> tensor @-> tensor @-> size_t @-> size_t @-> returning tensor)

  (** [cpy ctx a b] copies the data from tensor `a` to tensor `b`. Returns a view of `b`.
      @param ctx The context.
      @param a The source tensor.
      @param b The destination tensor.
      @return A view of the destination tensor `b`. *)
  let cpy = foreign (ns "cpy") (context @-> tensor @-> tensor @-> returning tensor)

  (** [cast ctx a typ] casts tensor `a` to the specified type `typ`.
      @param ctx The context.
      @param a The tensor to cast.
      @param typ The target type.
      @return The casted tensor. *)
  let cast = foreign (ns "cast") (context @-> tensor @-> typ @-> returning tensor)

  (** [cont ctx a] makes tensor `a` contiguous in memory.
      @param ctx The context.
      @param a The tensor.
      @return A contiguous version of the tensor `a`. *)
  let cont = foreign (ns "cont") (context @-> tensor @-> returning tensor)

  (** [cont_1d ctx a ne0] makes tensor `a` contiguous with a new 1D shape.
      @param ctx The context.
      @param a The tensor.
      @param ne0 The size of the first dimension.
      @return A contiguous 1D tensor. *)
  let cont_1d = foreign (ns "cont_1d") (context @-> tensor @-> int64_t @-> returning tensor)

  (** [cont_2d ctx a ne0 ne1] makes tensor `a` contiguous with a new 2D shape.
      @param ctx The context.
      @param a The tensor.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @return A contiguous 2D tensor. *)
  let cont_2d = foreign (ns "cont_2d") (context @-> tensor @-> int64_t @-> int64_t @-> returning tensor)

  (** [cont_3d ctx a ne0 ne1 ne2] makes tensor `a` contiguous with a new 3D shape.
      @param ctx The context.
      @param a The tensor.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @param ne2 The size of the third dimension.
      @return A contiguous 3D tensor. *)
  let cont_3d = foreign (ns "cont_3d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [cont_4d ctx a ne0 ne1 ne2 ne3] makes tensor `a` contiguous with a new 4D shape.
      @param ctx The context.
      @param a The tensor.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @param ne2 The size of the third dimension.
      @param ne3 The size of the fourth dimension.
      @return A contiguous 4D tensor. *)
  let cont_4d =
    foreign (ns "cont_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape ctx a b] creates a view of tensor `a` with the shape of tensor `b`.
      @param ctx The context.
      @param a The tensor to reshape.
      @param b Tensor defining the new shape.
      @return A view of `a` with the new shape. *)
  let reshape = foreign (ns "reshape") (context @-> tensor @-> tensor @-> returning tensor)

  (** [reshape_1d ctx a ne0] creates a 1D view of tensor `a`.
      @param ctx The context.
      @param a The tensor to reshape.
      @param ne0 The size of the first dimension.
      @return A 1D view of `a`. *)
  let reshape_1d = foreign (ns "reshape_1d") (context @-> tensor @-> int64_t @-> returning tensor)

  (** [reshape_2d ctx a ne0 ne1] creates a 2D view of tensor `a`.
      @param ctx The context.
      @param a The tensor to reshape.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @return A 2D view of `a`. *)
  let reshape_2d = foreign (ns "reshape_2d") (context @-> tensor @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape_3d ctx a ne0 ne1 ne2] creates a 3D view of tensor `a`.
      @param ctx The context.
      @param a The tensor to reshape.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @param ne2 The size of the third dimension.
      @return A 3D view of `a`. *)
  let reshape_3d =
    foreign (ns "reshape_3d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [reshape_4d ctx a ne0 ne1 ne2 ne3] creates a 4D view of tensor `a`.
      @param ctx The context.
      @param a The tensor to reshape.
      @param ne0 The size of the first dimension.
      @param ne1 The size of the second dimension.
      @param ne2 The size of the third dimension.
      @param ne3 The size of the fourth dimension.
      @return A 4D view of `a`. *)
  let reshape_4d =
    foreign (ns "reshape_4d") (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> returning tensor)

  (** [view_1d ctx a ne0 offset] creates a 1D view of tensor `a` starting at `offset`.
      @param ctx The context.
      @param a The source tensor.
      @param ne0 The size of the view's dimension.
      @param offset Offset in bytes from the start of `a`'s data.
      @return The 1D view tensor. *)
  let view_1d = foreign (ns "view_1d") (context @-> tensor @-> int64_t @-> size_t @-> returning tensor)

  (** [view_2d ctx a ne0 ne1 nb1 offset] creates a 2D view of tensor `a`.
      @param ctx The context.
      @param a The source tensor.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @param nb1 Row stride in bytes for the view.
      @param offset Offset in bytes from the start of `a`'s data.
      @return The 2D view tensor. *)
  let view_2d =
    foreign (ns "view_2d") (context @-> tensor @-> int64_t @-> int64_t @-> size_t @-> size_t @-> returning tensor)

  (** [view_3d ctx a ne0 ne1 ne2 nb1 nb2 offset] creates a 3D view of tensor `a`.
      @param ctx The context.
      @param a The source tensor.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @param ne2 Size of the third dimension.
      @param nb1 Row stride in bytes.
      @param nb2 Slice stride in bytes.
      @param offset Offset in bytes from the start of `a`'s data.
      @return The 3D view tensor. *)
  let view_3d =
    foreign (ns "view_3d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> returning tensor)

  (** [view_4d ctx a ne0 ne1 ne2 ne3 nb1 nb2 nb3 offset] creates a 4D view of tensor `a`.
      @param ctx The context.
      @param a The source tensor.
      @param ne0 Size of the first dimension.
      @param ne1 Size of the second dimension.
      @param ne2 Size of the third dimension.
      @param ne3 Size of the fourth dimension.
      @param nb1 Row stride in bytes.
      @param nb2 Slice stride in bytes.
      @param nb3 Stride for the fourth dimension in bytes.
      @param offset Offset in bytes from the start of `a`'s data.
      @return The 4D view tensor. *)
  let view_4d =
    foreign (ns "view_4d")
      (context @-> tensor @-> int64_t @-> int64_t @-> int64_t @-> int64_t @-> size_t @-> size_t @-> size_t @-> size_t
     @-> returning tensor)

  (** [permute ctx a axis0 axis1 axis2 axis3] permutes the dimensions of tensor `a`.
      @param ctx The context.
      @param a The tensor to permute.
      @param axis0 New index for the original dimension 0.
      @param axis1 New index for the original dimension 1.
      @param axis2 New index for the original dimension 2.
      @param axis3 New index for the original dimension 3.
      @return The permuted tensor (view). *)
  let permute = foreign (ns "permute") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [transpose ctx a] transposes the first two dimensions of tensor `a`. Alias for `permute(ctx, a, 1, 0, 2, 3)`.
      @param ctx The context.
      @param a The tensor to transpose.
      @return The transposed tensor (view). *)
  let transpose = foreign (ns "transpose") (context @-> tensor @-> returning tensor)

  (** [get_rows ctx a b] gathers rows from tensor `a` based on indices in tensor `b`. Supports 3D tensors where
      `a->ne[2] == b->ne[1]`.
      @param ctx The context.
      @param a The data tensor.
      @param b The tensor containing row indices.
      @return A tensor containing the gathered rows. *)
  let get_rows = foreign (ns "get_rows") (context @-> tensor @-> tensor @-> returning tensor)

  (** [get_rows_back ctx a b c] computes the backward pass for `get_rows`.
      @param ctx The context.
      @param a Gradient of the `get_rows` result.
      @param b Row indices used in the forward pass.
      @param c Original data tensor from the forward pass (used for shape).
      @return The gradient with respect to the original data tensor `a`. *)
  let get_rows_back = foreign (ns "get_rows_back") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [diag ctx a] creates a diagonal matrix from vector `a`, or extracts the diagonal from matrix `a`.
      @param ctx The context.
      @param a The input tensor (vector or matrix).
      @return The resulting diagonal matrix or vector. *)
  let diag = foreign (ns "diag") (context @-> tensor @-> returning tensor)

  (** [diag_mask_inf ctx a n_past] sets elements above the k-th diagonal (k = `n_past`) to -infinity.
      @param ctx The context.
      @param a The tensor to modify.
      @param n_past The diagonal offset (0 for main diagonal, >0 for upper diagonals).
      @return The modified tensor. *)
  let diag_mask_inf = foreign (ns "diag_mask_inf") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_inf_inplace ctx a n_past] sets elements above the k-th diagonal to -infinity in-place. Returns a view
      of `a`.
      @param ctx The context.
      @param a The tensor to modify (modified).
      @param n_past The diagonal offset.
      @return A view of the modified tensor `a`. *)
  let diag_mask_inf_inplace = foreign (ns "diag_mask_inf_inplace") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_zero ctx a n_past] sets elements above the k-th diagonal (k = `n_past`) to 0.
      @param ctx The context.
      @param a The tensor to modify.
      @param n_past The diagonal offset.
      @return The modified tensor. *)
  let diag_mask_zero = foreign (ns "diag_mask_zero") (context @-> tensor @-> int @-> returning tensor)

  (** [diag_mask_zero_inplace ctx a n_past] sets elements above the k-th diagonal to 0 in-place. Returns a view of `a`.
      @param ctx The context.
      @param a The tensor to modify (modified).
      @param n_past The diagonal offset.
      @return A view of the modified tensor `a`. *)
  let diag_mask_zero_inplace = foreign (ns "diag_mask_zero_inplace") (context @-> tensor @-> int @-> returning tensor)

  (** [soft_max ctx a] computes the softmax function along the first dimension (rows).
      @param ctx The context.
      @param a The input tensor.
      @return The tensor with softmax applied. *)
  let soft_max = foreign (ns "soft_max") (context @-> tensor @-> returning tensor)

  (** [soft_max_inplace ctx a] computes softmax along rows in-place. Returns a view of `a`.
      @param ctx The context.
      @param a The input tensor (modified).
      @return A view of the modified tensor `a`. *)
  let soft_max_inplace = foreign (ns "soft_max_inplace") (context @-> tensor @-> returning tensor)

  (** [soft_max_ext ctx a mask scale max_bias] computes fused softmax: `softmax(a*scale + mask*(ALiBi slope))`.
      @param ctx The context.
      @param a The input tensor.
      @param mask Optional mask tensor.
      @param scale Scaling factor for `a`.
      @param max_bias Maximum bias for ALiBi (0.0f for no ALiBi).
      @return The resulting tensor. *)
  let soft_max_ext = foreign (ns "soft_max_ext") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [soft_max_ext_back ctx a b scale max_bias] computes the backward pass for `soft_max_ext`.
      @param ctx The context.
      @param a Gradient of the `soft_max_ext` output.
      @param b Original output of the `soft_max_ext` forward pass.
      @param scale Scaling factor used in forward pass.
      @param max_bias Maximum bias used in forward pass.
      @return The gradient with respect to the input `a` of the forward pass. *)
  let soft_max_ext_back =
    foreign (ns "soft_max_ext_back") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [soft_max_ext_back_inplace ctx a b scale max_bias] computes the backward pass for `soft_max_ext` in-place. Returns
      a view of `a`.
      @param ctx The context.
      @param a Gradient tensor (modified).
      @param b Original output of the `soft_max_ext` forward pass.
      @param scale Scaling factor used in forward pass.
      @param max_bias Maximum bias used in forward pass.
      @return A view of the modified gradient tensor `a`. *)
  let soft_max_ext_back_inplace =
    foreign (ns "soft_max_ext_back_inplace") (context @-> tensor @-> tensor @-> float @-> float @-> returning tensor)

  (** [rope ctx a b n_dims mode] applies Rotary Positional Embedding (RoPE).
      @param ctx The context.
      @param a The input tensor.
      @param b Tensor containing positions (int32, size a->ne[2]).
      @param n_dims Number of dimensions to apply RoPE to.
      @param mode RoPE mode flags (e.g., `GGML_ROPE_TYPE_NEOX`).
      @return The tensor with RoPE applied. *)
  let rope = foreign (ns "rope") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [rope_inplace ctx a b n_dims mode] applies RoPE in-place. Returns a view of `a`.
      @param ctx The context.
      @param a The input tensor (modified).
      @param b Tensor containing positions.
      @param n_dims Number of dimensions for RoPE.
      @param mode RoPE mode flags.
      @return A view of the modified tensor `a`. *)
  let rope_inplace = foreign (ns "rope_inplace") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [rope_ext ctx a b c n_dims mode n_ctx_orig freq_base freq_scale ext_factor attn_factor beta_fast beta_slow]
      applies extended RoPE with custom parameters.
      @param ctx The context.
      @param a Input tensor.
      @param b Positions tensor.
      @param c Optional frequency factors tensor.
      @param n_dims Number of dimensions for RoPE.
      @param mode RoPE mode flags.
      @param n_ctx_orig Original context size for scaling (e.g., YaRN).
      @param freq_base Base frequency.
      @param freq_scale Frequency scaling factor.
      @param ext_factor Extrapolation factor (e.g., YaRN).
      @param attn_factor Attention scaling factor (e.g., YaRN).
      @param beta_fast Beta fast parameter (e.g., YaRN).
      @param beta_slow Beta slow parameter (e.g., YaRN).
      @return The tensor with extended RoPE applied. *)
  let rope_ext =
    foreign (ns "rope_ext")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_multi ctx a b c n_dims sections mode n_ctx_orig freq_base freq_scale ext_factor attn_factor beta_fast
       beta_slow] applies RoPE to multiple sections with different parameters.
      @param ctx The context.
      @param a Input tensor.
      @param b Positions tensor.
      @param c Optional frequency factors tensor.
      @param n_dims Number of dimensions for RoPE.
      @param sections Array defining the sections.
      @param mode RoPE mode flags.
      @param n_ctx_orig Original context size.
      @param freq_base Base frequency.
      @param freq_scale Frequency scaling factor.
      @param ext_factor Extrapolation factor.
      @param attn_factor Attention scaling factor.
      @param beta_fast Beta fast parameter.
      @param beta_slow Beta slow parameter.
      @return The tensor with multi-section RoPE applied. *)
  let rope_multi =
    foreign (ns "rope_multi")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> ptr int @-> int @-> int @-> float @-> float @-> float
     @-> float @-> float @-> float @-> returning tensor)

  (** [rope_ext_inplace ctx a b c n_dims mode ...] applies extended RoPE in-place. Returns a view of `a`. (Parameters
      same as `rope_ext`).
      @return A view of the modified tensor `a`. *)
  let rope_ext_inplace =
    foreign (ns "rope_ext_inplace")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_yarn_corr_dims n_dims n_ctx_orig freq_base beta_fast beta_slow dims] computes correction dimensions for YaRN
      RoPE scaling.
      @param n_dims Number of dimensions for RoPE.
      @param n_ctx_orig Original context size.
      @param freq_base Base frequency.
      @param beta_fast Beta fast parameter.
      @param beta_slow Beta slow parameter.
      @param dims Output pointer to store the two correction dimensions. *)
  let rope_yarn_corr_dims =
    foreign (ns "rope_yarn_corr_dims") (int @-> int @-> float @-> float @-> float @-> ptr float @-> returning void)

  (** [rope_ext_back ctx a b c n_dims mode ...] computes the backward pass for `rope_ext`. (Parameters mostly same as
      `rope_ext`, `a` is the gradient dy).
      @param a Gradient of the `rope_ext` output.
      @param b Positions tensor from forward pass.
      @param c Optional frequency factors tensor from forward pass.
      @return The gradient dx. *)
  let rope_ext_back =
    foreign (ns "rope_ext_back")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> int @-> int @-> float @-> float @-> float @-> float
     @-> float @-> float @-> returning tensor)

  (** [rope_multi_back ctx a b c n_dims sections mode ...] computes the backward pass for `rope_multi`. (Parameters
      mostly same as `rope_multi`, `a` is the gradient dy).
      @param a Gradient of the `rope_multi` output.
      @param b Positions tensor from forward pass.
      @param c Optional frequency factors tensor from forward pass.
      @return The gradient dx. *)
  let rope_multi_back =
    foreign (ns "rope_multi_back")
      (context @-> tensor @-> tensor @-> tensor @-> int @-> ptr int @-> int @-> int @-> float @-> float @-> float
     @-> float @-> float @-> float @-> returning tensor)

  (** [clamp ctx a min max] clamps the elements of tensor `a` between `min` and `max`. Returns a view of `a`.
      @param ctx The context.
      @param a The tensor to clamp (modified).
      @param min Minimum value.
      @param max Maximum value.
      @return A view of the modified tensor `a`. *)
  let clamp = foreign (ns "clamp") (context @-> tensor @-> float @-> float @-> returning tensor)

  (** [im2col ctx a b s0 s1 p0 p1 d0 d1 is_2D dst_type] implements the im2col operation used in convolutions.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0.
      @param p1 Padding dimension 1.
      @param d0 Dilation dimension 0.
      @param d1 Dilation dimension 1.
      @param is_2D Whether it's a 2D operation.
      @param dst_type The desired type for the output tensor.
      @return The resulting tensor after im2col transformation. *)
  let im2col =
    foreign (ns "im2col")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> bool @-> typ
     @-> returning tensor)

  (** [im2col_back ctx a b ne s0 s1 p0 p1 d0 d1 is_2D] computes the backward pass for `im2col`.
      @param ctx The context.
      @param a Convolution kernel from forward pass.
      @param b Gradient of the `im2col` output.
      @param ne Shape of the original input data (`b` in forward pass).
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0.
      @param p1 Padding dimension 1.
      @param d0 Dilation dimension 0.
      @param d1 Dilation dimension 1.
      @param is_2D Whether it was a 2D operation.
      @return The gradient with respect to the input data. *)
  let im2col_back =
    foreign (ns "im2col_back")
      (context @-> tensor @-> tensor @-> ptr int64_t @-> int @-> int @-> int @-> int @-> int @-> int @-> bool
     @-> returning tensor)

  (** [conv_1d ctx a b s0 p0 d0] performs 1D convolution.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride.
      @param p0 Padding.
      @param d0 Dilation.
      @return The result of the convolution. *)
  let conv_1d = foreign (ns "conv_1d") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_1d_ph ctx a b s d] performs 1D convolution with 'half' padding. Alias for `conv_1d(a, b, s, a->ne[0]/2, d)`.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s Stride.
      @param d Dilation.
      @return The result of the convolution. *)
  let conv_1d_ph = foreign (ns "conv_1d_ph") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [conv_1d_dw ctx a b s0 p0 d0] performs 1D depthwise convolution.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride.
      @param p0 Padding.
      @param d0 Dilation.
      @return The result of the depthwise convolution. *)
  let conv_1d_dw = foreign (ns "conv_1d_dw") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_1d_dw_ph ctx a b s0 d0] performs 1D depthwise convolution with 'half' padding.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride.
      @param d0 Dilation.
      @return The result of the depthwise convolution. *)
  let conv_1d_dw_ph = foreign (ns "conv_1d_dw_ph") (context @-> tensor @-> tensor @-> int @-> int @-> returning tensor)

  (** [conv_transpose_1d ctx a b s0 p0 d0] performs 1D transposed convolution (deconvolution).
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride.
      @param p0 Padding.
      @param d0 Dilation.
      @return The result of the transposed convolution. *)
  let conv_transpose_1d =
    foreign (ns "conv_transpose_1d") (context @-> tensor @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [conv_2d ctx a b s0 s1 p0 p1 d0 d1] performs 2D convolution.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0.
      @param p1 Padding dimension 1.
      @param d0 Dilation dimension 0.
      @param d1 Dilation dimension 1.
      @return The result of the 2D convolution. *)
  let conv_2d =
    foreign (ns "conv_2d")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [conv_2d_sk_p0 ctx a b] performs 2D convolution with stride equal to kernel size and zero padding.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @return The result of the convolution. *)
  let conv_2d_sk_p0 = foreign (ns "conv_2d_sk_p0") (context @-> tensor @-> tensor @-> returning tensor)

  (** [conv_2d_s1_ph ctx a b] performs 2D convolution with stride 1 and 'half' padding.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @return The result of the convolution. *)
  let conv_2d_s1_ph = foreign (ns "conv_2d_s1_ph") (context @-> tensor @-> tensor @-> returning tensor)

  (** [conv_2d_dw ctx a b s0 s1 p0 p1 d0 d1] performs 2D depthwise convolution.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0.
      @param p1 Padding dimension 1.
      @param d0 Dilation dimension 0.
      @param d1 Dilation dimension 1.
      @return The result of the 2D depthwise convolution. *)
  let conv_2d_dw =
    foreign (ns "conv_2d_dw")
      (context @-> tensor @-> tensor @-> int @-> int @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [conv_transpose_2d_p0 ctx a b stride] performs 2D transposed convolution with zero padding.
      @param ctx The context.
      @param a Convolution kernel.
      @param b Input data.
      @param stride The stride.
      @return The result of the transposed convolution. *)
  let conv_transpose_2d_p0 =
    foreign (ns "conv_transpose_2d_p0") (context @-> tensor @-> tensor @-> int @-> returning tensor)

  (** [pool_1d ctx a op k0 s0 p0] performs 1D pooling.
      @param ctx The context.
      @param a Input tensor.
      @param op Pooling operation type (`GGML_OP_POOL_MAX`, `GGML_OP_POOL_AVG`).
      @param k0 Kernel size.
      @param s0 Stride.
      @param p0 Padding.
      @return The result of the 1D pooling. *)
  let pool_1d = foreign (ns "pool_1d") (context @-> tensor @-> op_pool @-> int @-> int @-> int @-> returning tensor)

  (** [pool_2d ctx a op k0 k1 s0 s1 p0 p1] performs 2D pooling.
      @param ctx The context.
      @param a Input tensor.
      @param op Pooling operation type.
      @param k0 Kernel size dimension 0.
      @param k1 Kernel size dimension 1.
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0 (float for potential fractional padding).
      @param p1 Padding dimension 1 (float for potential fractional padding).
      @return The result of the 2D pooling. *)
  let pool_2d =
    foreign (ns "pool_2d")
      (context @-> tensor @-> op_pool @-> int @-> int @-> int @-> int @-> float @-> float @-> returning tensor)

  (** [pool_2d_back ctx a af op k0 k1 s0 s1 p0 p1] computes the backward pass for 2D pooling.
      @param ctx The context.
      @param a Gradient of the `pool_2d` output.
      @param af Original input tensor from the forward pass.
      @param op Pooling operation type used in forward pass.
      @param k0 Kernel size dimension 0.
      @param k1 Kernel size dimension 1.
      @param s0 Stride dimension 0.
      @param s1 Stride dimension 1.
      @param p0 Padding dimension 0.
      @param p1 Padding dimension 1.
      @return The gradient with respect to the input of the forward pass. *)
  let pool_2d_back =
    foreign (ns "pool_2d_back")
      (context @-> tensor @-> tensor @-> op_pool @-> int @-> int @-> int @-> int @-> float @-> float
     @-> returning tensor)

  (** [upscale ctx a scale_factor] performs nearest neighbor upscaling by `scale_factor` on the first two dimensions.
      @param ctx The context.
      @param a Input tensor.
      @param scale_factor The integer factor to scale dimensions by.
      @return The upscaled tensor. *)
  let upscale = foreign (ns "upscale") (context @-> tensor @-> int @-> returning tensor)

  (** [upscale_ext ctx a ne0 ne1 ne2 ne3] performs nearest neighbor upscaling to the specified dimensions.
      @param ctx The context.
      @param a Input tensor.
      @param ne0 Target size for dimension 0.
      @param ne1 Target size for dimension 1.
      @param ne2 Target size for dimension 2.
      @param ne3 Target size for dimension 3.
      @return The upscaled tensor. *)
  let upscale_ext = foreign (ns "upscale_ext") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [pad ctx a p0 p1 p2 p3] pads each dimension of tensor `a` with zeros.
      @param ctx The context.
      @param a Input tensor.
      @param p0 Padding for dimension 0.
      @param p1 Padding for dimension 1.
      @param p2 Padding for dimension 2.
      @param p3 Padding for dimension 3.
      @return The padded tensor. *)
  let pad = foreign (ns "pad") (context @-> tensor @-> int @-> int @-> int @-> int @-> returning tensor)

  (** [pad_reflect_1d ctx a p0 p1] pads the first two dimensions of tensor `a` using reflection padding.
      @param ctx The context.
      @param a Input tensor.
      @param p0 Padding for dimension 0.
      @param p1 Padding for dimension 1.
      @return The padded tensor. *)
  let pad_reflect_1d = foreign (ns "pad_reflect_1d") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [timestep_embedding ctx timesteps dim max_period] creates timestep embeddings used in diffusion models.
      @param ctx The context.
      @param timesteps Tensor of timesteps [N,].
      @param dim Embedding dimension.
      @param max_period Maximum period for the sinusoidal embedding.
      @return Tensor of embeddings [N, dim]. *)
  let timestep_embedding = foreign (ns "timestep_embedding") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [argsort ctx a order] returns the indices that would sort tensor `a` along rows.
      @param ctx The context.
      @param a Input tensor.
      @param order Sort order (`GGML_SORT_ORDER_ASC` or `GGML_SORT_ORDER_DESC`).
      @return Tensor containing the sorted indices. *)
  let argsort = foreign (ns "argsort") (context @-> tensor @-> sort_order @-> returning tensor)

  (** [arange ctx start stop step] creates a 1D tensor with values ranging from `start` to `stop` (exclusive) with
      `step`.
      @param ctx The context.
      @param start Start value.
      @param stop Stop value.
      @param step Step value.
      @return The 1D tensor containing the range. *)
  let arange = foreign (ns "arange") (context @-> float @-> float @-> float @-> returning tensor)

  (** [top_k ctx a k] returns the values and indices of the top `k` elements along the last dimension.
      @param ctx The context.
      @param a Input tensor.
      @param k The number of top elements to select.
      @return A tensor containing the top k values and indices (implementation specific). *)
  let top_k = foreign (ns "top_k") (context @-> tensor @-> int @-> returning tensor)

  (** [flash_attn_ext ctx q k v mask scale max_bias logit_softcap] performs extended Flash Attention. q:
      [n_embd_k, n_batch, n_head, 1], k: [n_embd_k, n_kv, n_head_kv, 1], v: [n_embd_v, n_kv, n_head_kv, 1], mask:
      [n_kv, n_batch_pad, 1, 1]
      @param ctx The context.
      @param q Query tensor.
      @param k Key tensor.
      @param v Value tensor (not transposed).
      @param mask Optional attention mask.
      @param scale Scaling factor for QK^T.
      @param max_bias Maximum bias for ALiBi.
      @param logit_softcap Softcap value for logits.
      @return Result tensor [n_embd_v, n_head, n_batch, 1] (permuted). *)
  let flash_attn_ext =
    foreign (ns "flash_attn_ext")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> float @-> float @-> float @-> returning tensor)

  (** [flash_attn_ext_set_prec a prec] sets the precision for the Flash Attention operation involving tensor `a`.
      @param a The tensor involved in the Flash Attention operation.
      @param prec The desired precision. *)
  let flash_attn_ext_set_prec = foreign (ns "flash_attn_ext_set_prec") (tensor @-> prec @-> returning void)

  (** [flash_attn_ext_get_prec a] gets the precision currently set for the Flash Attention operation involving tensor
      `a`.
      @param a The tensor involved in the Flash Attention operation.
      @return The current precision. *)
  let flash_attn_ext_get_prec = foreign (ns "flash_attn_ext_get_prec") (tensor @-> returning prec)

  (** [flash_attn_back ctx q k v d masked] computes the backward pass for Flash Attention. (Note: Needs adaptation for
      `flash_attn_ext`).
      @param ctx The context.
      @param q Query tensor from forward pass.
      @param k Key tensor from forward pass.
      @param v Value tensor from forward pass.
      @param d Gradient of the Flash Attention output.
      @param masked Whether masking was used in the forward pass.
      @return Gradient with respect to the input(s). *)
  let flash_attn_back =
    foreign (ns "flash_attn_back") (context @-> tensor @-> tensor @-> tensor @-> tensor @-> bool @-> returning tensor)

  (** [ssm_conv ctx sx c] performs Structured State Space Model (SSM) convolution.
      @param ctx The context.
      @param sx State tensor.
      @param c Convolution kernel.
      @return Result of the SSM convolution. *)
  let ssm_conv = foreign (ns "ssm_conv") (context @-> tensor @-> tensor @-> returning tensor)

  (** [ssm_scan ctx s x dt A B C] performs Structured State Space Model (SSM) scan.
      @param ctx The context.
      @param s State tensor.
      @param x Input tensor.
      @param dt Delta t tensor.
      @param A State transition matrix A.
      @param B State transition matrix B.
      @param C Output matrix C.
      @return Result of the SSM scan. *)
  let ssm_scan =
    foreign (ns "ssm_scan")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [win_part ctx a w] partitions tensor `a` into non-overlapping windows of size `w`.
      @param ctx The context.
      @param a Input tensor.
      @param w Window size.
      @return Tensor containing the window partitions. *)
  let win_part = foreign (ns "win_part") (context @-> tensor @-> int @-> returning tensor)

  (** [win_unpart ctx a w0 h0 w] reverses the window partitioning operation.
      @param ctx The context.
      @param a Tensor containing window partitions.
      @param w0 Original width before partitioning.
      @param h0 Original height before partitioning.
      @param w Window size used during partitioning.
      @return The reconstructed tensor. *)
  let win_unpart = foreign (ns "win_unpart") (context @-> tensor @-> int @-> int @-> int @-> returning tensor)

  (** [unary ctx a op] applies a unary operation `op` to tensor `a`.
      @param ctx The context.
      @param a Input tensor.
      @param op Unary operation type.
      @return The resulting tensor. *)
  let unary = foreign (ns "unary") (context @-> tensor @-> unary_op @-> returning tensor)

  (** [unary_inplace ctx a op] applies a unary operation `op` to tensor `a` in-place.
      @param ctx The context.
      @param a Input tensor (modified).
      @param op Unary operation type.
      @return The modified tensor `a`. *)
  let unary_inplace = foreign (ns "unary_inplace") (context @-> tensor @-> unary_op @-> returning tensor)

  (** [get_rel_pos ctx a qh kh] computes relative positional embeddings. Used in SAM.
      @param ctx The context.
      @param a Input tensor containing positional information.
      @param qh Query height/width.
      @param kh Key height/width.
      @return Tensor containing relative positional embeddings. *)
  let get_rel_pos = foreign (ns "get_rel_pos") (context @-> tensor @-> int @-> int @-> returning tensor)

  (** [add_rel_pos ctx a pw ph] adds relative positional embeddings to tensor `a`. Used in SAM.
      @param ctx The context.
      @param a Input tensor (e.g., attention scores).
      @param pw Relative position embedding for width.
      @param ph Relative position embedding for height.
      @return Tensor with added positional embeddings. *)
  let add_rel_pos = foreign (ns "add_rel_pos") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [add_rel_pos_inplace ctx a pw ph] adds relative positional embeddings to `a` in-place. Returns a view of `a`.
      @param ctx The context.
      @param a Input tensor (modified).
      @param pw Relative position embedding for width.
      @param ph Relative position embedding for height.
      @return A view of the modified tensor `a`. *)
  let add_rel_pos_inplace =
    foreign (ns "add_rel_pos_inplace") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [rwkv_wkv6 ctx k v r tf td state] computes the RWKV v6 WKV operation.
      @param ctx The context.
      @param k Key tensor.
      @param v Value tensor.
      @param r Receptance tensor.
      @param tf Time factor tensor.
      @param td Time decay tensor.
      @param state State tensor.
      @return Result of the WKV operation. *)
  let rwkv_wkv6 =
    foreign (ns "rwkv_wkv6")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [gated_linear_attn ctx k v q g state scale] computes Gated Linear Attention.
      @param ctx The context.
      @param k Key tensor.
      @param v Value tensor.
      @param q Query tensor.
      @param g Gate tensor.
      @param state State tensor.
      @param scale Scaling factor.
      @return Result of the Gated Linear Attention. *)
  let gated_linear_attn =
    foreign (ns "gated_linear_attn")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> float @-> returning tensor)

  (** [rwkv_wkv7 ctx r w k v a b state] computes the RWKV v7 WKV operation.
      @param ctx The context.
      @param r Receptance tensor.
      @param w Weight tensor.
      @param k Key tensor.
      @param v Value tensor.
      @param a Alpha tensor (state).
      @param b Beta tensor (state).
      @param state State tensor (previous state).
      @return Result of the WKV operation. *)
  let rwkv_wkv7 =
    foreign (ns "rwkv_wkv7")
      (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [quantize_init typ] initializes quantization resources for the given type.
      @param typ The quantization type. *)
  let quantize_init = foreign (ns "quantize_init") (typ @-> returning void)

  (** [quantize_free ()] frees quantization resources. *)
  let quantize_free = foreign (ns "quantize_free") (void @-> returning void)

  (** [quantize_requires_imatrix typ] checks if the quantization type requires an importance matrix.
      @param typ The quantization type.
      @return True if an importance matrix is required, false otherwise. *)
  let quantize_requires_imatrix = foreign (ns "quantize_requires_imatrix") (typ @-> returning bool)

  (** [quantize_chunk typ src dst start N num_threads imatrix] quantizes a chunk of data.
      @param typ Target quantization type.
      @param src Pointer to the source f32 data.
      @param dst Pointer to the destination quantized data buffer.
      @param start Starting index of the chunk.
      @param N Number of elements in the chunk.
      @param num_threads Number of threads to use (unused in current C impl).
      @param imatrix Optional importance matrix.
      @return Size of the quantized data in bytes. *)
  let quantize_chunk =
    foreign (ns "quantize_chunk")
      (typ @-> ptr float @-> ptr void @-> int64_t @-> int64_t @-> int64_t @-> ptr float @-> returning size_t)

  (** [cross_entropy_loss ctx a b] computes the cross-entropy loss between `a` (logits) and `b` (labels).
      @param ctx The context.
      @param a Logits tensor.
      @param b Labels tensor.
      @return Scalar tensor containing the loss. *)
  let cross_entropy_loss = foreign (ns "cross_entropy_loss") (context @-> tensor @-> tensor @-> returning tensor)

  (** [cross_entropy_loss_back ctx a b c] computes the backward pass for cross-entropy loss.
      @param ctx The context.
      @param a Gradient of the loss.
      @param b Logits tensor from the forward pass.
      @param c Labels tensor from the forward pass.
      @return Gradient with respect to the logits `a`. *)
  let cross_entropy_loss_back =
    foreign (ns "cross_entropy_loss_back") (context @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [opt_step_adamw ctx w dw m v hparams] performs an AdamW optimization step.
      @param ctx The context.
      @param w Weight tensor (modified).
      @param dw Gradient tensor.
      @param m First moment tensor (modified).
      @param v Second moment tensor (modified).
      @param hparams Hyperparameters tensor.
      @return The modified weight tensor `w`. *)
  let opt_step_adamw =
    foreign (ns "opt_step_adamw") (context @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> returning tensor)

  (** [build_forward_expand graph tensor] expands the forward graph to include the computation of `tensor`.
      @param graph The computation graph.
      @param tensor The tensor whose computation to include. *)
  let build_forward_expand = foreign (ns "build_forward_expand") (cgraph @-> tensor @-> returning void)

  (** [build_backward_expand ctx_fwd ctx_bwd graph keep_grads] expands the backward graph.
      @param ctx_fwd Forward context.
      @param ctx_bwd Backward context.
      @param graph The computation graph.
      @param keep_grads Whether to keep intermediate gradients. *)
  let build_backward_expand =
    foreign (ns "build_backward_expand") (context @-> context @-> cgraph @-> bool @-> returning void)

  (** [new_graph ctx] creates a new computation graph with the default size.
      @param ctx The context.
      @return The new computation graph. *)
  let new_graph = foreign (ns "new_graph") (context @-> returning cgraph)

  (** [new_graph_custom ctx size grads] creates a new computation graph with a custom size.
      @param ctx The context.
      @param size The maximum number of nodes in the graph.
      @param grads Whether the graph will store gradients.
      @return The new computation graph. *)
  let new_graph_custom = foreign (ns "new_graph_custom") (context @-> size_t @-> bool @-> returning cgraph)

  (** [graph_dup ctx graph] duplicates a computation graph.
      @param ctx The context.
      @param graph The graph to duplicate.
      @return The duplicated graph. *)
  let graph_dup = foreign (ns "graph_dup") (context @-> cgraph @-> returning cgraph)

  (** [graph_cpy src dst] copies the nodes from graph `src` to `dst`.
      @param src The source graph.
      @param dst The destination graph. *)
  let graph_cpy = foreign (ns "graph_cpy") (cgraph @-> cgraph @-> returning void)

  (** [graph_reset graph] resets the gradient data for all nodes in the graph.
      @param graph The computation graph. *)
  let graph_reset = foreign (ns "graph_reset") (cgraph @-> returning void)

  (** [graph_clear graph] clears the nodes from the graph.
      @param graph The computation graph. *)
  let graph_clear = foreign (ns "graph_clear") (cgraph @-> returning void)

  (** [graph_size graph] returns the number of nodes currently in the graph.
      @param graph The computation graph.
      @return The number of nodes. *)
  let graph_size = foreign (ns "graph_size") (cgraph @-> returning int)

  (** [graph_node graph i] returns the i-th tensor node in the graph.
      @param graph The computation graph.
      @param i The index of the node.
      @return The tensor node. *)
  let graph_node = foreign (ns "graph_node") (cgraph @-> int @-> returning tensor)

  (** [graph_nodes graph] returns a pointer to the array of tensor nodes in the graph.
      @param graph The computation graph.
      @return Pointer to the first tensor node. *)
  let graph_nodes = foreign (ns "graph_nodes") (cgraph @-> returning (ptr tensor))
  (* Returns ptr to the first tensor *)

  (** [graph_n_nodes graph] returns the number of nodes currently in the graph (same as `graph_size`).
      @param graph The computation graph.
      @return The number of nodes. *)
  let graph_n_nodes = foreign (ns "graph_n_nodes") (cgraph @-> returning int)

  (** [graph_add_node graph tensor] adds a tensor node to the graph. (Internal use likely).
      @param graph The computation graph.
      @param tensor The tensor node to add. *)
  let graph_add_node = foreign (ns "graph_add_node") (cgraph @-> tensor @-> returning void)

  (** [graph_overhead ()] returns the memory overhead of a default-sized graph structure.
      @return Overhead in bytes. *)
  let graph_overhead = foreign (ns "graph_overhead") (void @-> returning size_t)

  (** [graph_overhead_custom size grads] returns the memory overhead for a custom-sized graph.
      @param size The maximum number of nodes.
      @param grads Whether the graph stores gradients.
      @return Overhead in bytes. *)
  let graph_overhead_custom = foreign (ns "graph_overhead_custom") (size_t @-> bool @-> returning size_t)

  (** [graph_get_tensor graph name] retrieves a tensor from the graph by its name.
      @param graph The computation graph.
      @param name The name of the tensor.
      @return The tensor, or NULL if not found. *)
  let graph_get_tensor = foreign (ns "graph_get_tensor") (cgraph @-> string @-> returning tensor)

  (** [graph_get_grad graph tensor] retrieves the gradient tensor associated with a given tensor in the graph.
      @param graph The computation graph.
      @param tensor The tensor whose gradient is requested.
      @return The gradient tensor, or NULL if gradients are not stored or not computed. *)
  let graph_get_grad = foreign (ns "graph_get_grad") (cgraph @-> tensor @-> returning tensor)

  (** [graph_get_grad_acc graph tensor] retrieves the accumulated gradient tensor. (Likely internal use).
      @param graph The computation graph.
      @param tensor The tensor.
      @return The accumulated gradient tensor. *)
  let graph_get_grad_acc = foreign (ns "graph_get_grad_acc") (cgraph @-> tensor @-> returning tensor)

  (*
  (* Graph Import/Export *)
  (** [graph_export graph fname] exports the computation graph to a file.
      @param graph The graph to export.
      @param fname The filename. *)
  let graph_export = foreign (ns "graph_export") (cgraph @-> string @-> returning void)

  (** [graph_import fname ctx_fwd ctx_bwd] imports a computation graph from a file.
      @param fname The filename.
      @param ctx_fwd Pointer to store the loaded forward context.
      @param ctx_bwd Pointer to store the loaded backward context.
      @return The imported computation graph. *)
  let graph_import = foreign (ns "graph_import") (string @-> ptr context @-> ptr context @-> returning cgraph)
  *)

  (** [graph_print graph] prints information about the computation graph to stderr.
      @param graph The computation graph. *)
  let graph_print = foreign (ns "graph_print") (cgraph @-> returning void)

  (** [graph_dump_dot gf gb filename] dumps the computation graph(s) in DOT format to a file.
      @param gf Forward graph (optional).
      @param gb Backward graph (optional).
      @param filename The output filename. *)
  let graph_dump_dot = foreign (ns "graph_dump_dot") (cgraph @-> cgraph @-> string @-> returning void)

  (** [numa_init strategy] initializes NUMA (Non-Uniform Memory Access) support.
      @param strategy The NUMA strategy to use. *)
  let numa_init = foreign (ns "numa_init") (numa_strategy @-> returning void)

  (** [is_numa ()] checks if NUMA support is enabled.
      @return True if NUMA is enabled, false otherwise. *)
  let is_numa = foreign (ns "is_numa") (void @-> returning bool)

  (** [new_i32 ctx value] creates a new scalar tensor of type i32.
      @param ctx The context.
      @param value The int32 value.
      @return The new scalar tensor. *)
  let new_i32 = foreign (ns "new_i32") (context @-> int32_t @-> returning tensor)

  (** [new_f32 ctx value] creates a new scalar tensor of type f32.
      @param ctx The context.
      @param value The float value.
      @return The new scalar tensor. *)
  let new_f32 = foreign (ns "new_f32") (context @-> float @-> returning tensor)

  (** [set_i32 tensor value] sets the value of a scalar i32 tensor. Returns the tensor itself.
      @param tensor The scalar tensor (modified).
      @param value The int32 value to set.
      @return The modified tensor. *)
  let set_i32 = foreign (ns "set_i32") (tensor @-> int32_t @-> returning tensor)

  (** [set_f32 tensor value] sets the value of a scalar f32 tensor. Returns the tensor itself.
      @param tensor The scalar tensor (modified).
      @param value The float value to set.
      @return The modified tensor. *)
  let set_f32 = foreign (ns "set_f32") (tensor @-> float @-> returning tensor)

  (** [get_i32_1d tensor i] gets the i32 value at index `i` in a 1D tensor.
      @param tensor The 1D tensor.
      @param i The index.
      @return The int32 value. *)
  let get_i32_1d = foreign (ns "get_i32_1d") (tensor @-> int @-> returning int32_t)

  (** [set_i32_1d tensor i value] sets the i32 value at index `i` in a 1D tensor.
      @param tensor The 1D tensor (modified).
      @param i The index.
      @param value The int32 value to set. *)
  let set_i32_1d = foreign (ns "set_i32_1d") (tensor @-> int @-> int32_t @-> returning void)

  (** [get_i32_nd tensor i0 i1 i2 i3] gets the i32 value at the specified multi-dimensional index.
      @param tensor The tensor.
      @param i0 Index for dimension 0.
      @param i1 Index for dimension 1.
      @param i2 Index for dimension 2.
      @param i3 Index for dimension 3.
      @return The int32 value. *)
  let get_i32_nd = foreign (ns "get_i32_nd") (tensor @-> int @-> int @-> int @-> int @-> returning int32_t)

  (** [set_i32_nd tensor i0 i1 i2 i3 value] sets the i32 value at the specified multi-dimensional index.
      @param tensor The tensor (modified).
      @param i0 Index for dimension 0.
      @param i1 Index for dimension 1.
      @param i2 Index for dimension 2.
      @param i3 Index for dimension 3.
      @param value The int32 value to set. *)
  let set_i32_nd = foreign (ns "set_i32_nd") (tensor @-> int @-> int @-> int @-> int @-> int32_t @-> returning void)

  (** [get_f32_1d tensor i] gets the float value at index `i` in a 1D tensor.
      @param tensor The 1D tensor.
      @param i The index.
      @return The float value. *)
  let get_f32_1d = foreign (ns "get_f32_1d") (tensor @-> int @-> returning float)

  (** [set_f32_1d tensor i value] sets the float value at index `i` in a 1D tensor.
      @param tensor The 1D tensor (modified).
      @param i The index.
      @param value The float value to set. *)
  let set_f32_1d = foreign (ns "set_f32_1d") (tensor @-> int @-> float @-> returning void)

  (** [get_f32_nd tensor i0 i1 i2 i3] gets the float value at the specified multi-dimensional index.
      @param tensor The tensor.
      @param i0 Index for dimension 0.
      @param i1 Index for dimension 1.
      @param i2 Index for dimension 2.
      @param i3 Index for dimension 3.
      @return The float value. *)
  let get_f32_nd = foreign (ns "get_f32_nd") (tensor @-> int @-> int @-> int @-> int @-> returning float)

  (** [set_f32_nd tensor i0 i1 i2 i3 value] sets the float value at the specified multi-dimensional index.
      @param tensor The tensor (modified).
      @param i0 Index for dimension 0.
      @param i1 Index for dimension 1.
      @param i2 Index for dimension 2.
      @param i3 Index for dimension 3.
      @param value The float value to set. *)
  let set_f32_nd = foreign (ns "set_f32_nd") (tensor @-> int @-> int @-> int @-> int @-> float @-> returning void)

  (** [threadpool_new params] creates a new threadpool.
      @param params Threadpool parameters (currently unused, pass NULL).
      @return Pointer to the new threadpool. *)
  let threadpool_new = foreign (ns "threadpool_new") (ptr void @-> returning (ptr threadpool))

  (** [threadpool_free pool] frees the resources associated with a threadpool.
      @param pool Pointer to the threadpool to free. *)
  let threadpool_free = foreign (ns "threadpool_free") (ptr threadpool @-> returning void)

  (*
  (** [threadpool_get_n_threads pool] gets the number of threads in the pool.
      @param pool Pointer to the threadpool.
      @return Number of threads. *)
  let threadpool_get_n_threads = foreign (ns "threadpool_get_n_threads") (ptr threadpool @-> returning int)
  *)

  (** [threadpool_pause pool] pauses the threads in the threadpool.
      @param pool Pointer to the threadpool. *)
  let threadpool_pause = foreign (ns "threadpool_pause") (ptr threadpool @-> returning void)

  (** [threadpool_resume pool] resumes the threads in the threadpool.
      @param pool Pointer to the threadpool. *)
  let threadpool_resume = foreign (ns "threadpool_resume") (ptr threadpool @-> returning void)

  (** [graph_plan graph n_threads pool] creates a computation plan for the graph.
      @param graph The computation graph.
      @param n_threads Number of threads to use (overrides pool setting if pool is NULL).
      @param pool Optional threadpool to use.
      @return The computation plan. *)
  let graph_plan = foreign (ns "graph_plan") (cgraph @-> int @-> ptr threadpool @-> returning Cplan.t)

  (** [graph_compute graph plan] computes the graph according to the plan.
      @param graph The computation graph.
      @param plan Pointer to the computation plan.
      @return Computation status (`GGML_STATUS_SUCCESS` on success). *)
  let graph_compute = foreign (ns "graph_compute") (cgraph @-> ptr Cplan.t @-> returning status)

  (** [graph_compute_with_ctx ctx graph n_threads] computes the graph using the provided context and number of threads.
      @param ctx The context (used for backend selection, e.g., CPU).
      @param graph The computation graph.
      @param n_threads Number of threads to use.
      @return Computation status. *)
  let graph_compute_with_ctx = foreign (ns "graph_compute_with_ctx") (context @-> cgraph @-> int @-> returning status)

  (** [cpu_has_sse3 () ] checks if the CPU supports SSE3 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_sse3 = foreign (ns "cpu_has_sse3") (void @-> returning int)

  (** [cpu_has_ssse3 ()] checks if the CPU supports SSSE3 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_ssse3 = foreign (ns "cpu_has_ssse3") (void @-> returning int)

  (** [cpu_has_avx ()] checks if the CPU supports AVX instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx = foreign (ns "cpu_has_avx") (void @-> returning int)

  (** [cpu_has_avx_vnni ()] checks if the CPU supports AVX VNNI instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx_vnni = foreign (ns "cpu_has_avx_vnni") (void @-> returning int)

  (** [cpu_has_avx2 ()] checks if the CPU supports AVX2 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx2 = foreign (ns "cpu_has_avx2") (void @-> returning int)

  (** [cpu_has_bmi2 ()] checks if the CPU supports BMI2 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_bmi2 = foreign (ns "cpu_has_bmi2") (void @-> returning int)

  (** [cpu_has_f16c ()] checks if the CPU supports F16C (half-precision conversion) instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_f16c = foreign (ns "cpu_has_f16c") (void @-> returning int)

  (** [cpu_has_fma ()] checks if the CPU supports FMA (fused multiply-add) instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_fma = foreign (ns "cpu_has_fma") (void @-> returning int)

  (** [cpu_has_avx512 ()] checks if the CPU supports AVX512 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx512 = foreign (ns "cpu_has_avx512") (void @-> returning int)

  (** [cpu_has_avx512_vbmi ()] checks if the CPU supports AVX512 VBMI instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_vbmi = foreign (ns "cpu_has_avx512_vbmi") (void @-> returning int)

  (** [cpu_has_avx512_vnni ()] checks if the CPU supports AVX512 VNNI instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_vnni = foreign (ns "cpu_has_avx512_vnni") (void @-> returning int)

  (** [cpu_has_avx512_bf16 ()] checks if the CPU supports AVX512 BF16 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_bf16 = foreign (ns "cpu_has_avx512_bf16") (void @-> returning int)

  (** [cpu_has_amx_int8 ()] checks if the CPU supports AMX INT8 instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_amx_int8 = foreign (ns "cpu_has_amx_int8") (void @-> returning int)

  (** [cpu_has_neon ()] checks if the CPU supports NEON instructions (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_neon = foreign (ns "cpu_has_neon") (void @-> returning int)

  (** [cpu_has_arm_fma ()] checks if the CPU supports ARM FMA instructions.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_arm_fma = foreign (ns "cpu_has_arm_fma") (void @-> returning int)

  (** [cpu_has_fp16_va ()] checks if the CPU supports FP16 vector arithmetic (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_fp16_va = foreign (ns "cpu_has_fp16_va") (void @-> returning int)

  (** [cpu_has_dotprod ()] checks if the CPU supports Dot Product instructions (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_dotprod = foreign (ns "cpu_has_dotprod") (void @-> returning int)

  (** [cpu_has_matmul_int8 ()] checks if the CPU supports INT8 matrix multiplication instructions (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_matmul_int8 = foreign (ns "cpu_has_matmul_int8") (void @-> returning int)

  (** [cpu_has_sve ()] checks if the CPU supports SVE instructions (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_sve = foreign (ns "cpu_has_sve") (void @-> returning int)

  (** [cpu_get_sve_cnt ()] gets the SVE vector length.
      @return SVE vector length in bits. *)
  let cpu_get_sve_cnt = foreign (ns "cpu_get_sve_cnt") (void @-> returning int)

  (** [cpu_has_sme ()] checks if the CPU supports SME instructions (ARM).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_sme = foreign (ns "cpu_has_sme") (void @-> returning int)

  (** [cpu_has_riscv_v ()] checks if the CPU supports RISC-V Vector extension.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_riscv_v = foreign (ns "cpu_has_riscv_v") (void @-> returning int)

  (** [cpu_has_vsx ()] checks if the CPU supports VSX instructions (PowerPC).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_vsx = foreign (ns "cpu_has_vsx") (void @-> returning int)

  (** [cpu_has_vxe ()] checks if the CPU supports VXE instructions (PowerPC).
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_vxe = foreign (ns "cpu_has_vxe") (void @-> returning int)

  (** [cpu_has_wasm_simd ()] checks if the environment supports WASM SIMD.
      @return 1 if supported, 0 otherwise. *)
  let cpu_has_wasm_simd = foreign (ns "cpu_has_wasm_simd") (void @-> returning int)

  (** [cpu_has_llamafile ()] checks if running within llamafile environment.
      @return 1 if true, 0 otherwise. *)
  let cpu_has_llamafile = foreign (ns "cpu_has_llamafile") (void @-> returning int)

  (** [get_type_traits_cpu typ] gets the CPU-specific type traits for a given type.
      @param typ The ggml type.
      @return Pointer to the constant type traits structure. *)
  let get_type_traits_cpu = foreign (ns "get_type_traits_cpu") (typ @-> returning (ptr @@ const TypeTraitsCpu.t))

  (** [cpu_init ()] initializes CPU-specific features (e.g., detects capabilities). *)
  let cpu_init = foreign (ns "cpu_init") (void @-> returning void)

  (** [backend_cpu_init ()] initializes the CPU backend.
      @return The CPU backend handle. *)
  let backend_cpu_init = foreign (ns "backend_cpu_init") (void @-> returning backend_t)

  (** [backend_is_cpu backend] checks if the given backend is the CPU backend.
      @param backend The backend handle.
      @return True if it's the CPU backend, false otherwise. *)
  let backend_is_cpu = foreign (ns "backend_is_cpu") (backend_t @-> returning bool)

  (** [backend_cpu_set_n_threads backend n_threads] sets the number of threads for the CPU backend.
      @param backend The CPU backend handle.
      @param n_threads The desired number of threads. *)
  let backend_cpu_set_n_threads = foreign (ns "backend_cpu_set_n_threads") (backend_t @-> int @-> returning void)

  (** [backend_cpu_set_threadpool backend pool] sets a custom threadpool for the CPU backend.
      @param backend The CPU backend handle.
      @param pool Pointer to the threadpool. *)
  let backend_cpu_set_threadpool =
    foreign (ns "backend_cpu_set_threadpool") (backend_t @-> ptr threadpool @-> returning void)

  (** [backend_cpu_set_abort_callback backend callback data] sets an abort callback for the CPU backend.
      @param backend The CPU backend handle.
      @param callback The callback function.
      @param data User data to pass to the callback. *)
  let backend_cpu_set_abort_callback =
    foreign (ns "backend_cpu_set_abort_callback") (backend_t @-> abort_callback @-> ptr void @-> returning void)

  (** [backend_cpu_reg ()] gets the registration information for the CPU backend.
      @return The backend registration structure. *)
  let backend_cpu_reg = foreign (ns "backend_cpu_reg") (void @-> returning backend_reg_t)

  module GGUF = struct
    open Types.GGUF

    let ns name = "gguf_" ^ name

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
        @param only_meta
          If true, only write metadata (header, KV, tensor info); if false, write metadata and tensor data.
        @return True on success, false on failure. *)
    let write_to_file = foreign (ns "write_to_file") (context_t @-> string @-> bool @-> returning bool)

    (** [get_meta_size ctx] calculates the size in bytes of the metadata section (including padding).
        @param ctx The GGUF context.
        @return The size of the metadata in bytes. *)
    let get_meta_size = foreign (ns "get_meta_size") (context_t @-> returning size_t)

    (** [get_meta_data ctx data] writes the metadata section to the provided buffer.
        @param ctx The GGUF context.
        @param data A void pointer to a buffer large enough to hold the metadata (size obtained from `get_meta_size`).
    *)
    let get_meta_data = foreign (ns "get_meta_data") (context_t @-> ptr void @-> returning void)
  end
end
