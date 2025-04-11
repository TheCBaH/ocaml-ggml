module Status = struct
  type t = AllocFailed | Failed | Success | Aborted

  let values = [ (AllocFailed, "ALLOC_FAILED"); (Failed, "FAILED"); (Success, "SUCCESS"); (Aborted, "ABORTED") ]
end

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
end

module Prec = struct
  type t = Default | F32

  let values = [ (Default, "DEFAULT"); (F32, "F32") ]
end

module Ftype = struct
  type t =
    | Unknown
    | All_F32
    | Mostly_F16
    | Mostly_Q4_0
    | Mostly_Q4_1
    | Mostly_Q4_1_Some_F16
    | Mostly_Q8_0
    | Mostly_Q5_0
    | Mostly_Q5_1
    | Mostly_Q2_K
    | Mostly_Q3_K
    | Mostly_Q4_K
    | Mostly_Q5_K
    | Mostly_Q6_K
    | Mostly_IQ2_XXS
    | Mostly_IQ2_XS
    | Mostly_IQ3_XXS
    | Mostly_IQ1_S
    | Mostly_IQ4_NL
    | Mostly_IQ3_S
    | Mostly_IQ2_S
    | Mostly_IQ4_XS
    | Mostly_IQ1_M
    | Mostly_BF16

  let values =
    [
      (Unknown, "UNKNOWN");
      (All_F32, "ALL_F32");
      (Mostly_F16, "MOSTLY_F16");
      (Mostly_Q4_0, "MOSTLY_Q4_0");
      (Mostly_Q4_1, "MOSTLY_Q4_1");
      (Mostly_Q4_1_Some_F16, "MOSTLY_Q4_1_SOME_F16");
      (Mostly_Q8_0, "MOSTLY_Q8_0");
      (Mostly_Q5_0, "MOSTLY_Q5_0");
      (Mostly_Q5_1, "MOSTLY_Q5_1");
      (Mostly_Q2_K, "MOSTLY_Q2_K");
      (Mostly_Q3_K, "MOSTLY_Q3_K");
      (Mostly_Q4_K, "MOSTLY_Q4_K");
      (Mostly_Q5_K, "MOSTLY_Q5_K");
      (Mostly_Q6_K, "MOSTLY_Q6_K");
      (Mostly_IQ2_XXS, "MOSTLY_IQ2_XXS");
      (Mostly_IQ2_XS, "MOSTLY_IQ2_XS");
      (Mostly_IQ3_XXS, "MOSTLY_IQ3_XXS");
      (Mostly_IQ1_S, "MOSTLY_IQ1_S");
      (Mostly_IQ4_NL, "MOSTLY_IQ4_NL");
      (Mostly_IQ3_S, "MOSTLY_IQ3_S");
      (Mostly_IQ2_S, "MOSTLY_IQ2_S");
      (Mostly_IQ4_XS, "MOSTLY_IQ4_XS");
      (Mostly_IQ1_M, "MOSTLY_IQ1_M");
      (Mostly_BF16, "MOSTLY_BF16");
    ]
end

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
    | Sum_Rows
    | Mean
    | Argmax
    | Count_Equal
    | Repeat
    | Repeat_Back
    | Concat
    | Silu_Back
    | Norm
    | Rms_Norm
    | Rms_Norm_Back
    | Group_Norm
    | L2_Norm
    | Mul_Mat
    | Mul_Mat_Id
    | Out_Prod
    | Scale
    | Set
    | Cpy
    | Cont
    | Reshape
    | View
    | Permute
    | Transpose
    | Get_Rows
    | Get_Rows_Back
    | Diag
    | Diag_Mask_Inf
    | Diag_Mask_Zero
    | Soft_Max
    | Soft_Max_Back
    | Rope
    | Rope_Back
    | Clamp
    | Conv_Transpose_1D
    | Im2Col
    | Im2Col_Back
    | Conv_Transpose_2D
    | Pool_1D
    | Pool_2D
    | Pool_2D_Back
    | Upscale
    | Pad
    | Pad_Reflect_1D
    | Arange
    | Timestep_Embedding
    | Argsort
    | Leaky_Relu
    | Flash_Attn_Ext
    | Flash_Attn_Back
    | Ssm_Conv
    | Ssm_Scan
    | Win_Part
    | Win_Unpart
    | Get_Rel_Pos
    | Add_Rel_Pos
    | Rwkv_Wkv6
    | Gated_Linear_Attn
    | Rwkv_Wkv7
    | Unary
    | Map_Unary
    | Map_Binary
    | Map_Custom1_F32
    | Map_Custom2_F32
    | Map_Custom3_F32
    | Map_Custom1
    | Map_Custom2
    | Map_Custom3
    | Cross_Entropy_Loss
    | Cross_Entropy_Loss_Back
    | Opt_Step_Adamw
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
      (Sum_Rows, "SUM_ROWS");
      (Mean, "MEAN");
      (Argmax, "ARGMAX");
      (Count_Equal, "COUNT_EQUAL");
      (Repeat, "REPEAT");
      (Repeat_Back, "REPEAT_BACK");
      (Concat, "CONCAT");
      (Silu_Back, "SILU_BACK");
      (Norm, "NORM");
      (Rms_Norm, "RMS_NORM");
      (Rms_Norm_Back, "RMS_NORM_BACK");
      (Group_Norm, "GROUP_NORM");
      (L2_Norm, "L2_NORM");
      (Mul_Mat, "MUL_MAT");
      (Mul_Mat_Id, "MUL_MAT_ID");
      (Out_Prod, "OUT_PROD");
      (Scale, "SCALE");
      (Set, "SET");
      (Cpy, "CPY");
      (Cont, "CONT");
      (Reshape, "RESHAPE");
      (View, "VIEW");
      (Permute, "PERMUTE");
      (Transpose, "TRANSPOSE");
      (Get_Rows, "GET_ROWS");
      (Get_Rows_Back, "GET_ROWS_BACK");
      (Diag, "DIAG");
      (Diag_Mask_Inf, "DIAG_MASK_INF");
      (Diag_Mask_Zero, "DIAG_MASK_ZERO");
      (Soft_Max, "SOFT_MAX");
      (Soft_Max_Back, "SOFT_MAX_BACK");
      (Rope, "ROPE");
      (Rope_Back, "ROPE_BACK");
      (Clamp, "CLAMP");
      (Conv_Transpose_1D, "CONV_TRANSPOSE_1D");
      (Im2Col, "IM2COL");
      (Im2Col_Back, "IM2COL_BACK");
      (Conv_Transpose_2D, "CONV_TRANSPOSE_2D");
      (Pool_1D, "POOL_1D");
      (Pool_2D, "POOL_2D");
      (Pool_2D_Back, "POOL_2D_BACK");
      (Upscale, "UPSCALE");
      (Pad, "PAD");
      (Pad_Reflect_1D, "PAD_REFLECT_1D");
      (Arange, "ARANGE");
      (Timestep_Embedding, "TIMESTEP_EMBEDDING");
      (Argsort, "ARGSORT");
      (Leaky_Relu, "LEAKY_RELU");
      (Flash_Attn_Ext, "FLASH_ATTN_EXT");
      (Flash_Attn_Back, "FLASH_ATTN_BACK");
      (Ssm_Conv, "SSM_CONV");
      (Ssm_Scan, "SSM_SCAN");
      (Win_Part, "WIN_PART");
      (Win_Unpart, "WIN_UNPART");
      (Get_Rel_Pos, "GET_REL_POS");
      (Add_Rel_Pos, "ADD_REL_POS");
      (Rwkv_Wkv6, "RWKV_WKV6");
      (Gated_Linear_Attn, "GATED_LINEAR_ATTN");
      (Rwkv_Wkv7, "RWKV_WKV7");
      (Unary, "UNARY");
      (Map_Unary, "MAP_UNARY");
      (Map_Binary, "MAP_BINARY");
      (Map_Custom1_F32, "MAP_CUSTOM1_F32");
      (Map_Custom2_F32, "MAP_CUSTOM2_F32");
      (Map_Custom3_F32, "MAP_CUSTOM3_F32");
      (Map_Custom1, "MAP_CUSTOM1");
      (Map_Custom2, "MAP_CUSTOM2");
      (Map_Custom3, "MAP_CUSTOM3");
      (Cross_Entropy_Loss, "CROSS_ENTROPY_LOSS");
      (Cross_Entropy_Loss_Back, "CROSS_ENTROPY_LOSS_BACK");
      (Opt_Step_Adamw, "OPT_STEP_ADAMW");
      (Count, "COUNT");
    ]
end

module Unary_op = struct
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
    | Gelu_Quick
    | Silu
    | Hardswish
    | Hardsigmoid
    | Exp
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
      (Gelu_Quick, "GELU_QUICK");
      (Silu, "SILU");
      (Hardswish, "HARDSWISH");
      (Hardsigmoid, "HARDSIGMOID");
      (Exp, "EXP");
      (Count, "COUNT");
    ]
end

module Object_type = struct
  type t = Tensor | Graph | Work_Buffer

  let values = [ (Tensor, "TENSOR"); (Graph, "GRAPH"); (Work_Buffer, "WORK_BUFFER") ]
end

module Log_level = struct
  type t = None | Debug | Info | Warn | Error | Cont

  let values = [ (None, "NONE"); (Debug, "DEBUG"); (Info, "INFO"); (Warn, "WARN"); (Error, "ERROR"); (Cont, "CONT") ]
end

module Tensor_flag = struct
  type t = Input | Output | Param | Loss

  let values = [ (Input, "INPUT"); (Output, "OUTPUT"); (Param, "PARAM"); (Loss, "LOSS") ]
end

module Op_pool = struct
  type t = Max | Avg | Count

  let values = [ (Max, "MAX"); (Avg, "AVG"); (Count, "COUNT") ]
end

module Sort_order = struct
  type t = Asc | Desc

  let values = [ (Asc, "ASC"); (Desc, "DESC") ]
end
