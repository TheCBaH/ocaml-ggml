(** Vocabulary types. *)
module VocabType = struct
  type t = None | Spm | Bpe | Wpm | Ugm | Rwkv

  let values = [ (None, "NONE"); (Spm, "SPM"); (Bpe, "BPE"); (Wpm, "WPM"); (Ugm, "UGM"); (Rwkv, "RWKV") ]
  let to_string t = List.assoc t values
end

(** Pre-tokenization types. *)
module VocabPreType = struct
  type t =
    | Default
    | Llama3
    | Deepseek_Llm
    | Deepseek_Coder
    | Falcon
    | Mpt
    | Starcoder
    | Gpt2
    | Refact
    | Command_R
    | Stablelm2
    | Qwen2
    | Olmo
    | Dbrx
    | Smaug
    | Poro
    | Chatglm3
    | Chatglm4
    | Viking
    | Jais
    | Tekken
    | Smollm
    | Codeshell
    | Bloom
    | Gpt3_Finnish
    | Exaone
    | Chameleon
    | Minerva
    | Deepseek3_Llm
    | Gpt4o
    | Superbpe
    | Trillion
    | Bailingmoe
    | Llama4

  let values =
    [
      (Default, "DEFAULT");
      (Llama3, "LLAMA3");
      (Deepseek_Llm, "DEEPSEEK_LLM");
      (Deepseek_Coder, "DEEPSEEK_CODER");
      (Falcon, "FALCON");
      (Mpt, "MPT");
      (Starcoder, "STARCODER");
      (Gpt2, "GPT2");
      (Refact, "REFACT");
      (Command_R, "COMMAND_R");
      (Stablelm2, "STABLELM2");
      (Qwen2, "QWEN2");
      (Olmo, "OLMO");
      (Dbrx, "DBRX");
      (Smaug, "SMAUG");
      (Poro, "PORO");
      (Chatglm3, "CHATGLM3");
      (Chatglm4, "CHATGLM4");
      (Viking, "VIKING");
      (Jais, "JAIS");
      (Tekken, "TEKKEN");
      (Smollm, "SMOLLM");
      (Codeshell, "CODESHELL");
      (Bloom, "BLOOM");
      (Gpt3_Finnish, "GPT3_FINNISH");
      (Exaone, "EXAONE");
      (Chameleon, "CHAMELEON");
      (Minerva, "MINERVA");
      (Deepseek3_Llm, "DEEPSEEK3_LLM");
      (Gpt4o, "GPT4O");
      (Superbpe, "SUPERBPE");
      (Trillion, "TRILLION");
      (Bailingmoe, "BAILINGMOE");
      (Llama4, "LLAMA4");
    ]

  let to_string t = List.assoc t values
end

(** RoPE types. *)
module RopeType = struct
  type t = None | Norm | Neox | Mrope | Vision

  let values =
    [
      (None, "NONE");
      (Norm, "NORM");
      (Neox, "NEOX");
      (* GGML_ROPE_TYPE_NEOX *)
      (Mrope, "MROPE");
      (* GGML_ROPE_TYPE_MROPE *)
      (Vision, "VISION");
      (* GGML_ROPE_TYPE_VISION *)
    ]

  let to_string t = List.assoc t values
end

(** Token types (deprecated). *)
module TokenType = struct
  type t = Undefined | Normal | Unknown | Control | User_Defined | Unused | Byte

  let values =
    [
      (Undefined, "UNDEFINED");
      (Normal, "NORMAL");
      (Unknown, "UNKNOWN");
      (Control, "CONTROL");
      (User_Defined, "USER_DEFINED");
      (Unused, "UNUSED");
      (Byte, "BYTE");
    ]

  let to_string t = List.assoc t values
end

(** Token attributes (bit flags). *)
module TokenAttr = struct
  type t =
    | Undefined
    | Unknown
    | Unused
    | Normal
    | Control
    | User_Defined
    | Byte
    | Normalized
    | Lstrip
    | Rstrip
    | Single_Word

  let values =
    [
      (Undefined, "UNDEFINED");
      (Unknown, "UNKNOWN");
      (Unused, "UNUSED");
      (Normal, "NORMAL");
      (Control, "CONTROL");
      (User_Defined, "USER_DEFINED");
      (Byte, "BYTE");
      (Normalized, "NORMALIZED");
      (Lstrip, "LSTRIP");
      (Rstrip, "RSTRIP");
      (Single_Word, "SINGLE_WORD");
    ]

  let to_string t = List.assoc t values
  (* Note: These are bit flags in C, consider using a list or set for multiple attributes *)
end

(** Model file types. *)
module Ftype = struct
  type t =
    | All_F32
    | Mostly_F16
    | Mostly_Q4_0
    | Mostly_Q4_1
    | Mostly_Q8_0
    | Mostly_Q5_0
    | Mostly_Q5_1
    | Mostly_Q2_K
    | Mostly_Q3_K_S
    | Mostly_Q3_K_M
    | Mostly_Q3_K_L
    | Mostly_Q4_K_S
    | Mostly_Q4_K_M
    | Mostly_Q5_K_S
    | Mostly_Q5_K_M
    | Mostly_Q6_K
    | Mostly_IQ2_XXS
    | Mostly_IQ2_XS
    | Mostly_Q2_K_S
    | Mostly_IQ3_XS
    | Mostly_IQ3_XXS
    | Mostly_IQ1_S
    | Mostly_IQ4_NL
    | Mostly_IQ3_S
    | Mostly_IQ3_M
    | Mostly_IQ2_S
    | Mostly_IQ2_M
    | Mostly_IQ4_XS
    | Mostly_IQ1_M
    | Mostly_BF16
    | Mostly_TQ1_0
    | Mostly_TQ2_0
    | Guessed

  let values =
    [
      (All_F32, "ALL_F32");
      (Mostly_F16, "MOSTLY_F16");
      (Mostly_Q4_0, "MOSTLY_Q4_0");
      (Mostly_Q4_1, "MOSTLY_Q4_1");
      (Mostly_Q8_0, "MOSTLY_Q8_0");
      (Mostly_Q5_0, "MOSTLY_Q5_0");
      (Mostly_Q5_1, "MOSTLY_Q5_1");
      (Mostly_Q2_K, "MOSTLY_Q2_K");
      (Mostly_Q3_K_S, "MOSTLY_Q3_K_S");
      (Mostly_Q3_K_M, "MOSTLY_Q3_K_M");
      (Mostly_Q3_K_L, "MOSTLY_Q3_K_L");
      (Mostly_Q4_K_S, "MOSTLY_Q4_K_S");
      (Mostly_Q4_K_M, "MOSTLY_Q4_K_M");
      (Mostly_Q5_K_S, "MOSTLY_Q5_K_S");
      (Mostly_Q5_K_M, "MOSTLY_Q5_K_M");
      (Mostly_Q6_K, "MOSTLY_Q6_K");
      (Mostly_IQ2_XXS, "MOSTLY_IQ2_XXS");
      (Mostly_IQ2_XS, "MOSTLY_IQ2_XS");
      (Mostly_Q2_K_S, "MOSTLY_Q2_K_S");
      (Mostly_IQ3_XS, "MOSTLY_IQ3_XS");
      (Mostly_IQ3_XXS, "MOSTLY_IQ3_XXS");
      (Mostly_IQ1_S, "MOSTLY_IQ1_S");
      (Mostly_IQ4_NL, "MOSTLY_IQ4_NL");
      (Mostly_IQ3_S, "MOSTLY_IQ3_S");
      (Mostly_IQ3_M, "MOSTLY_IQ3_M");
      (Mostly_IQ2_S, "MOSTLY_IQ2_S");
      (Mostly_IQ2_M, "MOSTLY_IQ2_M");
      (Mostly_IQ4_XS, "MOSTLY_IQ4_XS");
      (Mostly_IQ1_M, "MOSTLY_IQ1_M");
      (Mostly_BF16, "MOSTLY_BF16");
      (Mostly_TQ1_0, "MOSTLY_TQ1_0");
      (Mostly_TQ2_0, "MOSTLY_TQ2_0");
      (Guessed, "GUESSED");
    ]

  let to_string t = List.assoc t values
end

(** RoPE scaling types. *)
module RopeScalingType = struct
  type t = Unspecified | None | Linear | Yarn | Longrope | Max_Value

  let values =
    [
      (Unspecified, "UNSPECIFIED");
      (None, "NONE");
      (Linear, "LINEAR");
      (Yarn, "YARN");
      (Longrope, "LONGROPE");
      (Max_Value, "MAX_VALUE");
    ]

  let to_string t = List.assoc t values
end

(** Pooling types. *)
module PoolingType = struct
  type t = Unspecified | None | Mean | Cls | Last | Rank

  let values =
    [ (Unspecified, "UNSPECIFIED"); (None, "NONE"); (Mean, "MEAN"); (Cls, "CLS"); (Last, "LAST"); (Rank, "RANK") ]

  let to_string t = List.assoc t values
end

(** Attention types. *)
module AttentionType = struct
  type t = Unspecified | Causal | Non_Causal

  let values = [ (Unspecified, "UNSPECIFIED"); (Causal, "CAUSAL"); (Non_Causal, "NON_CAUSAL") ]
  let to_string t = List.assoc t values
end

(** Model splitting modes. *)
module SplitMode = struct
  type t = None | Layer | Row

  let values = [ (None, "NONE"); (Layer, "LAYER"); (Row, "ROW") ]
  let to_string t = List.assoc t values
end

(** Model key-value override types. *)
module ModelKvOverrideType = struct
  type t = Int | Float | Bool | Str

  let values = [ (Int, "INT"); (Float, "FLOAT"); (Bool, "BOOL"); (Str, "STR") ]
  let to_string t = List.assoc t values
end
