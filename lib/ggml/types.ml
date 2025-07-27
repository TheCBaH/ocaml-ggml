(** Status codes for ggml functions. *)
module Status = struct
  type t = AllocFailed | Failed | Success | Aborted

  let values = [ (AllocFailed, "ALLOC_FAILED"); (Failed, "FAILED"); (Success, "SUCCESS"); (Aborted, "ABORTED") ]
  let to_string t = List.assoc t values
end

(** Available tensor types. NOTE: always add types at the end of the enum to keep backward compatibility. *)
module Type = struct
  type t =
    | F32
    | F16
    | Q4_0
    | Q4_1
    | Q5_0
    | Q5_1
    | Q8_0
    | Q8_1
    | Q2_K
    | Q3_K
    | Q4_K
    | Q5_K
    | Q6_K
    | Q8_K
    | IQ2_XXS
    | IQ2_XS
    | IQ3_XXS
    | IQ1_S
    | IQ4_NL
    | IQ3_S
    | IQ2_S
    | IQ4_XS
    | I8
    | I16
    | I32
    | I64
    | F64
    | IQ1_M
    | BF16
    | TQ1_0
    | TQ2_0

  let values =
    [
      (F32, "F32");
      (F16, "F16");
      (Q4_0, "Q4_0");
      (Q4_1, "Q4_1");
      (Q5_0, "Q5_0");
      (Q5_1, "Q5_1");
      (Q8_0, "Q8_0");
      (Q8_1, "Q8_1");
      (Q2_K, "Q2_K");
      (Q3_K, "Q3_K");
      (Q4_K, "Q4_K");
      (Q5_K, "Q5_K");
      (Q6_K, "Q6_K");
      (Q8_K, "Q8_K");
      (IQ2_XXS, "IQ2_XXS");
      (IQ2_XS, "IQ2_XS");
      (IQ3_XXS, "IQ3_XXS");
      (IQ1_S, "IQ1_S");
      (IQ4_NL, "IQ4_NL");
      (IQ3_S, "IQ3_S");
      (IQ2_S, "IQ2_S");
      (IQ4_XS, "IQ4_XS");
      (I8, "I8");
      (I16, "I16");
      (I32, "I32");
      (I64, "I64");
      (F64, "F64");
      (IQ1_M, "IQ1_M");
      (BF16, "BF16");
      (TQ1_0, "TQ1_0");
      (TQ2_0, "TQ2_0");
    ]

  let to_string t = List.assoc t values
end

(** Precision levels for matrix multiplication. *)
module Prec = struct
  type t = Default | F32

  let values = [ (Default, "DEFAULT"); (F32, "F32") ]
  let to_string t = List.assoc t values
end

(** Model file types. *)
module Ftype = struct
  type t =
    | Unknown
    | AllF32
    | MostlyF16  (** except 1d tensors *)
    | MostlyQ4_0  (** except 1d tensors *)
    | MostlyQ4_1  (** except 1d tensors *)
    | MostlyQ4_1SomeF16  (** tok_embeddings.weight and output.weight are F16 *)
    | MostlyQ8_0  (** except 1d tensors *)
    | MostlyQ5_0  (** except 1d tensors *)
    | MostlyQ5_1  (** except 1d tensors *)
    | MostlyQ2_K  (** except 1d tensors *)
    | MostlyQ3_K  (** except 1d tensors *)
    | MostlyQ4_K  (** except 1d tensors *)
    | MostlyQ5_K  (** except 1d tensors *)
    | MostlyQ6_K  (** except 1d tensors *)
    | MostlyIQ2_XXS  (** except 1d tensors *)
    | MostlyIQ2_XS  (** except 1d tensors *)
    | MostlyIQ3_XXS  (** except 1d tensors *)
    | MostlyIQ1_S  (** except 1d tensors *)
    | MostlyIQ4_NL  (** except 1d tensors *)
    | MostlyIQ3_S  (** except 1d tensors *)
    | MostlyIQ2_S  (** except 1d tensors *)
    | MostlyIQ4_XS  (** except 1d tensors *)
    | MostlyIQ1_M  (** except 1d tensors *)
    | MostlyBF16  (** except 1d tensors *)

  let values =
    [
      (Unknown, "UNKNOWN");
      (AllF32, "ALL_F32");
      (MostlyF16, "MOSTLY_F16");
      (MostlyQ4_0, "MOSTLY_Q4_0");
      (MostlyQ4_1, "MOSTLY_Q4_1");
      (MostlyQ4_1SomeF16, "MOSTLY_Q4_1_SOME_F16");
      (MostlyQ8_0, "MOSTLY_Q8_0");
      (MostlyQ5_0, "MOSTLY_Q5_0");
      (MostlyQ5_1, "MOSTLY_Q5_1");
      (MostlyQ2_K, "MOSTLY_Q2_K");
      (MostlyQ3_K, "MOSTLY_Q3_K");
      (MostlyQ4_K, "MOSTLY_Q4_K");
      (MostlyQ5_K, "MOSTLY_Q5_K");
      (MostlyQ6_K, "MOSTLY_Q6_K");
      (MostlyIQ2_XXS, "MOSTLY_IQ2_XXS");
      (MostlyIQ2_XS, "MOSTLY_IQ2_XS");
      (MostlyIQ3_XXS, "MOSTLY_IQ3_XXS");
      (MostlyIQ1_S, "MOSTLY_IQ1_S");
      (MostlyIQ4_NL, "MOSTLY_IQ4_NL");
      (MostlyIQ3_S, "MOSTLY_IQ3_S");
      (MostlyIQ2_S, "MOSTLY_IQ2_S");
      (MostlyIQ4_XS, "MOSTLY_IQ4_XS");
      (MostlyIQ1_M, "MOSTLY_IQ1_M");
      (MostlyBF16, "MOSTLY_BF16");
    ]

  let to_string t = List.assoc t values
end

(** Available tensor operations. *)
module Op = struct
  type t =
    | None
    | Dup
    | Add
    | Add1
    | Acc
    | Sub
    | Mul
    | Div
    | Sqr
    | Sqrt
    | Log
    | Sin
    | Cos
    | Sum
    | SumRows
    | Mean
    | Argmax
    | CountEqual
    | Repeat
    | RepeatBack
    | Concat
    | SiluBack
    | Norm  (** normalize *)
    | RmsNorm
    | RmsNormBack
    | GroupNorm
    | L2Norm
    | MulMat
    | MulMatId
    | OutProd
    | Scale
    | Set
    | Cpy
    | Cont
    | Reshape
    | View
    | Permute
    | Transpose
    | GetRows
    | GetRowsBack
    | SetRows
    | Diag
    | DiagMaskInf
    | DiagMaskZero
    | SoftMax
    | SoftMaxBack
    | Rope
    | RopeBack
    | Clamp
    | ConvTranspose1D
    | Im2Col
    | Im2ColBack
    | Conv2d
    | Conv2dDw
    | ConvTranspose2D
    | Pool1D
    | Pool2D
    | Pool2DBack
    | Upscale  (** nearest interpolate *)
    | Pad
    | PadReflect1D
    | Roll
    | Arange
    | TimestepEmbedding
    | Argsort
    | LeakyRelu
    | FlashAttnExt
    | FlashAttnBack
    | SsmConv
    | SsmScan
    | WinPart
    | WinUnpart
    | GetRelPos
    | AddRelPos
    | RwkvWkv6
    | GatedLinearAttn
    | RwkvWkv7
    | Unary
    | MapCustom1
    | MapCustom2
    | MapCustom3
    | Custom
    | CrossEntropyLoss
    | CrossEntropyLossBack
    | OptStepAdamw
    | Glu
    | Count

  let values =
    [
      (None, "NONE");
      (Dup, "DUP");
      (Add, "ADD");
      (Add1, "ADD1");
      (Acc, "ACC");
      (Sub, "SUB");
      (Mul, "MUL");
      (Div, "DIV");
      (Sqr, "SQR");
      (Sqrt, "SQRT");
      (Log, "LOG");
      (Sin, "SIN");
      (Cos, "COS");
      (Sum, "SUM");
      (SumRows, "SUM_ROWS");
      (Mean, "MEAN");
      (Argmax, "ARGMAX");
      (CountEqual, "COUNT_EQUAL");
      (Repeat, "REPEAT");
      (RepeatBack, "REPEAT_BACK");
      (Concat, "CONCAT");
      (SiluBack, "SILU_BACK");
      (Norm, "NORM");
      (RmsNorm, "RMS_NORM");
      (RmsNormBack, "RMS_NORM_BACK");
      (GroupNorm, "GROUP_NORM");
      (L2Norm, "L2_NORM");
      (MulMat, "MUL_MAT");
      (MulMatId, "MUL_MAT_ID");
      (OutProd, "OUT_PROD");
      (Scale, "SCALE");
      (Set, "SET");
      (Cpy, "CPY");
      (Cont, "CONT");
      (Reshape, "RESHAPE");
      (View, "VIEW");
      (Permute, "PERMUTE");
      (Transpose, "TRANSPOSE");
      (GetRows, "GET_ROWS");
      (GetRowsBack, "GET_ROWS_BACK");
      (SetRows, "SET_ROWS");
      (Diag, "DIAG");
      (DiagMaskInf, "DIAG_MASK_INF");
      (DiagMaskZero, "DIAG_MASK_ZERO");
      (SoftMax, "SOFT_MAX");
      (SoftMaxBack, "SOFT_MAX_BACK");
      (Rope, "ROPE");
      (RopeBack, "ROPE_BACK");
      (Clamp, "CLAMP");
      (ConvTranspose1D, "CONV_TRANSPOSE_1D");
      (Im2Col, "IM2COL");
      (Im2ColBack, "IM2COL_BACK");
      (Conv2d, "CONV_2D");
      (Conv2dDw, "CONV_2D_DW");
      (ConvTranspose2D, "CONV_TRANSPOSE_2D");
      (Pool1D, "POOL_1D");
      (Pool2D, "POOL_2D");
      (Pool2DBack, "POOL_2D_BACK");
      (Upscale, "UPSCALE");
      (Pad, "PAD");
      (PadReflect1D, "PAD_REFLECT_1D");
      (Roll, "ROLL");
      (Arange, "ARANGE");
      (TimestepEmbedding, "TIMESTEP_EMBEDDING");
      (Argsort, "ARGSORT");
      (LeakyRelu, "LEAKY_RELU");
      (FlashAttnExt, "FLASH_ATTN_EXT");
      (FlashAttnBack, "FLASH_ATTN_BACK");
      (SsmConv, "SSM_CONV");
      (SsmScan, "SSM_SCAN");
      (WinPart, "WIN_PART");
      (WinUnpart, "WIN_UNPART");
      (GetRelPos, "GET_REL_POS");
      (AddRelPos, "ADD_REL_POS");
      (RwkvWkv6, "RWKV_WKV6");
      (GatedLinearAttn, "GATED_LINEAR_ATTN");
      (RwkvWkv7, "RWKV_WKV7");
      (Unary, "UNARY");
      (MapCustom1, "MAP_CUSTOM1");
      (MapCustom2, "MAP_CUSTOM2");
      (MapCustom3, "MAP_CUSTOM3");
      (Custom, "CUSTOM");
      (CrossEntropyLoss, "CROSS_ENTROPY_LOSS");
      (CrossEntropyLossBack, "CROSS_ENTROPY_LOSS_BACK");
      (OptStepAdamw, "OPT_STEP_ADAMW");
      (Glu, "GLU");
      (Count, "COUNT");
    ]

  let to_string t = List.assoc t values
end

(** Available unary operations. *)
module UnaryOp = struct
  type t =
    | Abs
    | Sgn
    | Neg
    | Step
    | Tanh
    | Elu
    | Relu
    | Sigmoid
    | Gelu
    | GeluQuick
    | Silu
    | Hardswish
    | Hardsigmoid
    | Exp
    | GeluErf
    | Count

  let values =
    [
      (Abs, "ABS");
      (Sgn, "SGN");
      (Neg, "NEG");
      (Step, "STEP");
      (Tanh, "TANH");
      (Elu, "ELU");
      (Relu, "RELU");
      (Sigmoid, "SIGMOID");
      (Gelu, "GELU");
      (GeluQuick, "GELU_QUICK");
      (Silu, "SILU");
      (Hardswish, "HARDSWISH");
      (Hardsigmoid, "HARDSIGMOID");
      (Exp, "EXP");
      (GeluErf, "GELU_ERF");
      (Count, "COUNT");
    ]

  let to_string t = List.assoc t values
end

(** Gated Linear Unit operations. *)
module GluOp = struct
  type t = Reglu | Geglu | Swiglu | GegluErf | GegluQuick | Count

  let values =
    [
      (Reglu, "REGLU");
      (Geglu, "GEGLU");
      (Swiglu, "SWIGLU");
      (GegluErf, "GEGLU_ERF");
      (GegluQuick, "GEGLU_QUICK");
      (Count, "COUNT");
    ]

  let to_string t = List.assoc t values
end

(** Object types used by ggml. *)
module ObjectType = struct
  type t = Tensor | Graph | WorkBuffer

  let values = [ (Tensor, "TENSOR"); (Graph, "GRAPH"); (WorkBuffer, "WORK_BUFFER") ]
  let to_string t = List.assoc t values
end

(** Logging levels. *)
module LogLevel = struct
  type t = None | Debug | Info | Warn | Error | Cont  (** continue previous log *)

  let values = [ (None, "NONE"); (Debug, "DEBUG"); (Info, "INFO"); (Warn, "WARN"); (Error, "ERROR"); (Cont, "CONT") ]
  let to_string t = List.assoc t values
end

(** Tensor flags. Used to mark tensors with special properties. *)
module TensorFlag = struct
  type t =
    | Input  (** is an input for the GGML compute graph *)
    | Output  (** is an output for the GGML compute graph *)
    | Param  (** contains trainable parameters *)
    | Loss  (** defines loss for numerical optimization (multiple loss tensors add up) *)

  let values = [ (Input, "INPUT"); (Output, "OUTPUT"); (Param, "PARAM"); (Loss, "LOSS") ]
  let to_string t = List.assoc t values
end

(** Pooling operations. *)
module OpPool = struct
  type t = Max | Avg | Count

  let values = [ (Max, "MAX"); (Avg, "AVG"); (Count, "COUNT") ]
  let to_string t = List.assoc t values
end

(** Sort order for argsort. *)
module SortOrder = struct
  type t = Asc | Desc

  let values = [ (Asc, "ASC"); (Desc, "DESC") ]
  let to_string t = List.assoc t values
end

(** NUMA strategy. *)
module NumaStrategy = struct
  type t = Disabled | Distribute | Isolate | Numactl | Mirror | Count

  let values =
    [
      (Disabled, "DISABLED");
      (Distribute, "DISTRIBUTE");
      (Isolate, "ISOLATE");
      (Numactl, "NUMACTL");
      (Mirror, "MIRROR");
      (Count, "COUNT");
    ]

  let to_string t = List.assoc t values
end

(** Scaling modes for upscale operations. *)
module ScaleMode = struct
  type t = Nearest | Bilinear | Count

  let values = [ (Nearest, "NEAREST"); (Bilinear, "BILINEAR"); (Count, "COUNT") ]
  let to_string t = List.assoc t values
end

(** Scaling flags for interpolate operations. *)
module ScaleFlag = struct
  type t = AlignCorners

  let values = [ (AlignCorners, "ALIGN_CORNERS") ]
  let to_string t = List.assoc t values
end

module Backend = struct
  (** Backend buffer usage types. *)
  module BufferUsage = struct
    type t = Any | Weights | Compute

    let values = [ (Any, "ANY"); (Weights, "WEIGHTS"); (Compute, "COMPUTE") ]
    let to_string t = List.assoc t values
  end

  (** Backend device types. *)
  module DevType = struct
    type t = Cpu | Gpu | Accel

    let values = [ (Cpu, "CPU"); (Gpu, "GPU"); (Accel, "ACCEL") ]
    let to_string t = List.assoc t values
  end
end

module GGUF = struct
  (** GGUF metadata value types. *)
  module Type = struct
    type t =
      | Uint8
      | Int8
      | Uint16
      | Int16
      | Uint32
      | Int32
      | Float32
      | Bool
      | String
      | Array
      | Uint64
      | Int64
      | Float64
      | Count  (** marks the end of the enum *)

    let values =
      [
        (Uint8, "UINT8");
        (Int8, "INT8");
        (Uint16, "UINT16");
        (Int16, "INT16");
        (Uint32, "UINT32");
        (Int32, "INT32");
        (Float32, "FLOAT32");
        (Bool, "BOOL");
        (String, "STRING");
        (Array, "ARRAY");
        (Uint64, "UINT64");
        (Int64, "INT64");
        (Float64, "FLOAT64");
        (Count, "COUNT");
      ]

    let to_string t = List.assoc t values
  end
end

module Opt = struct
  module BuildType = struct
    type t = Forward | Grad | Opt

    let values = [ (Forward, "FORWARD"); (Grad, "GRAD"); (Opt, "OPT") ]
    let to_string t = List.assoc t values
  end

  module LossType = struct
    type t = Mean | Sum | CrossEntropy | MeanSquaredError

    let values =
      [ (Mean, "MEAN"); (Sum, "SUM"); (CrossEntropy, "CROSS_ENTROPY"); (MeanSquaredError, "MEAN_SQUARED_ERROR") ]

    let to_string t = List.assoc t values
  end
end

(** Scheduling priorities. *)
module SchedPrio = struct
  type t = Low | Normal | Medium | High

  let values = [ (Low, "LOW"); (Normal, "NORMAL"); (Medium, "MEDIUM"); (High, "HIGH") ]
  let to_string t = List.assoc t values
end
