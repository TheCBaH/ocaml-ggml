module Status : sig
  type t = AllocFailed | Failed | Success | Aborted

  val values : (t * string) list
end

module Type : sig
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

  val values : (t * string) list
end

module Prec : sig
  type t = Default | F32

  val values : (t * string) list
end

module Ftype : sig
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

  val values : (t * string) list
end

module Op : sig
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

  val values : (t * string) list
end

module UnaryOp : sig
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

  val values : (t * string) list
end

module ObjectType : sig
  type t = Tensor | Graph | Work_Buffer

  val values : (t * string) list
end

module LogLevel : sig
  type t = None | Debug | Info | Warn | Error | Cont

  val values : (t * string) list
end

module TensorFlag : sig
  type t = Input | Output | Param | Loss

  val values : (t * string) list
end

module OpPool : sig
  type t = Max | Avg | Count

  val values : (t * string) list
end

module SortOrder : sig
  type t = Asc | Desc

  val values : (t * string) list
end

module NumaStrategy : sig
  type t = Disabled | Distribute | Isolate | Numactl | Mirror | Count

  val values : (t * string) list
end
