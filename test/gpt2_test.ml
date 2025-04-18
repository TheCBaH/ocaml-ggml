open Ctypes
open Gpt_2.C
open Model_explorer

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)
let attr key value = KeyValue.create ~key ~value

let pp_int64 fmt t = Format.fprintf fmt "%Ld" t

let pp_shape fmt tensor =
  let open Ggml.C in
  let rec cut_aux l' l =
    match l with
    | [] -> l'
    | hd::_ when hd = 1L || hd = 0L -> l'
    | hd::tl -> cut_aux (hd::l') tl in
  let ne = List.rev @@ cut_aux [] @@ CArray.to_list @@ getfp tensor Types.Tensor.ne in
  Format.(fprintf fmt "[%a]" (pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",@ ") pp_int64) ne)

let tensor n t =
  let name = getfp t Ggml.C.Types.Tensor.name in
  let name = to_string @@ CArray.start name in
  let name =
    let printed_name = if String.length name == 0 || String.starts_with ~prefix:"leaf_" name || String.starts_with ~prefix:"node_" name then ""
    else name ^ " " in
    Printf.sprintf "%s(%s)" printed_name @@ Ggml.C.Functions.type_name @@ getfp t Ggml.C.Types.Tensor.typ_ in
  let tensor_name = attr "tensor_name" name in
  let tensor_index = attr "tensor_index" @@ string_of_int n in
  let tensor_shape =
    let ne = getfp t Ggml.C.Types.Tensor.ne in
    let dims =
    if Ggml.C.Functions.is_matrix t then Printf.sprintf "[%Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1)
    else Printf.sprintf "[%Ld, %Ld, %Ld]" (CArray.get ne 0) (CArray.get ne 1) (CArray.get ne 2) in
    Printf.sprintf "%d %s %s" n dims (Ggml.C.Functions.op_symbol @@ getfp t Ggml.C.Types.Tensor.op) in
  ignore (tensor_name,tensor_index);
  ignore tensor_shape;
  ignore name;
  tensor_shape

let print pp t = Format.printf "@[%a@]@.%!" pp t

let%expect_test "gpt2" =
  let model = make Types.model_t in
  let n_ctx = 1024 in
  let n_gpu_layers = 0 in
  let fname = "models/gpt-2-117M/ggml-model.bin" in
  let rc = Functions.model_init (addr model) fname n_ctx n_gpu_layers in
  assert (rc = 0);
  [%expect {| gpt2_model_load: using CPU backend |}];

  let n_past = 0 in
  let n_tokens = 768 in

  let gpt2 = Functions.model_graph (addr model) n_past n_tokens in
  assert (not @@ Ctypes.is_null gpt2);
  let nodes = Ggml.C.Functions.graph_n_nodes gpt2 in
  Format.printf "nodes:%u" nodes;
  [%expect "nodes:487" ];

  print pp_shape @@ Ggml.C.Functions.graph_node gpt2 0;
  [%expect "[768, 768]" ];
  print pp_shape @@ Ggml.C.Functions.graph_node gpt2 6;
  [%expect "[2304, 768]" ];

  let nodes = Array.init nodes (fun n -> Ggml.C.Functions.graph_node gpt2 n) in
  let names = Array.map (fun t -> Ggml.C.Functions.op_name @@ getfp t Ggml.C.Types.Tensor.op) nodes in
  ignore names;
  let names = Array.mapi tensor nodes in

  Format.printf "@[%a@]" (Format.pp_print_list ~pp_sep:Format.pp_print_newline Format.pp_print_string)
  @@ Array.to_list names;
  [%expect
    {|
    0 [768, 768] get_rows(x)
    1 [768, 768] get_rows(x)
    2 [768, 768] x+y
    3 [768, 768] norm(x)
    4 [768, 768] x*y
    5 [768, 768] x+y
    6 [2304, 768] X*Y
    7 [2304, 768] x+y
    8 [768, 768] view(x)
    9 [589824, 1] view(x)
    10 [589824, 1] x-\>y
    11 [768, 768] view(x)
    12 [589824, 1] view(x)
    13 [589824, 1] x-\>y
    14 [589824, 1] view(x)
    15 [64, 12, 768] reshape(x)
    16 [768, 64, 12] permute(x)
    17 [768, 64, 12] cont(x)
    18 [589824, 1] view(x)
    19 [64, 12, 768] reshape(x)
    20 [64, 768, 12] permute(x)
    21 [768, 768] view(x)
    22 [64, 12, 768] cont(x)
    23 [64, 768, 12] permute(x)
    24 [768, 768, 12] X*Y
    25 [768, 768, 12] x*v
    26 [768, 768, 12] diag_mask_inf(x)
    27 [768, 768, 12] soft_max(x)
    28 [64, 768, 12] X*Y
    29 [64, 12, 768] permute(x)
    30 [768, 768] cont(x)
    31 [768, 768] X*Y
    32 [768, 768] x+y
    33 [768, 768] x+y
    34 [768, 768] norm(x)
    35 [768, 768] x*y
    36 [768, 768] x+y
    37 [3072, 768] X*Y
    38 [3072, 768] x+y
    39 [3072, 768] unary(x)
    40 [768, 768] X*Y
    41 [768, 768] x+y
    42 [768, 768] x+y
    43 [768, 768] norm(x)
    44 [768, 768] x*y
    45 [768, 768] x+y
    46 [2304, 768] X*Y
    47 [2304, 768] x+y
    48 [768, 768] view(x)
    49 [589824, 1] view(x)
    50 [589824, 1] x-\>y
    51 [768, 768] view(x)
    52 [589824, 1] view(x)
    53 [589824, 1] x-\>y
    54 [589824, 1] view(x)
    55 [64, 12, 768] reshape(x)
    56 [768, 64, 12] permute(x)
    57 [768, 64, 12] cont(x)
    58 [589824, 1] view(x)
    59 [64, 12, 768] reshape(x)
    60 [64, 768, 12] permute(x)
    61 [768, 768] view(x)
    62 [64, 12, 768] cont(x)
    63 [64, 768, 12] permute(x)
    64 [768, 768, 12] X*Y
    65 [768, 768, 12] x*v
    66 [768, 768, 12] diag_mask_inf(x)
    67 [768, 768, 12] soft_max(x)
    68 [64, 768, 12] X*Y
    69 [64, 12, 768] permute(x)
    70 [768, 768] cont(x)
    71 [768, 768] X*Y
    72 [768, 768] x+y
    73 [768, 768] x+y
    74 [768, 768] norm(x)
    75 [768, 768] x*y
    76 [768, 768] x+y
    77 [3072, 768] X*Y
    78 [3072, 768] x+y
    79 [3072, 768] unary(x)
    80 [768, 768] X*Y
    81 [768, 768] x+y
    82 [768, 768] x+y
    83 [768, 768] norm(x)
    84 [768, 768] x*y
    85 [768, 768] x+y
    86 [2304, 768] X*Y
    87 [2304, 768] x+y
    88 [768, 768] view(x)
    89 [589824, 1] view(x)
    90 [589824, 1] x-\>y
    91 [768, 768] view(x)
    92 [589824, 1] view(x)
    93 [589824, 1] x-\>y
    94 [589824, 1] view(x)
    95 [64, 12, 768] reshape(x)
    96 [768, 64, 12] permute(x)
    97 [768, 64, 12] cont(x)
    98 [589824, 1] view(x)
    99 [64, 12, 768] reshape(x)
    100 [64, 768, 12] permute(x)
    101 [768, 768] view(x)
    102 [64, 12, 768] cont(x)
    103 [64, 768, 12] permute(x)
    104 [768, 768, 12] X*Y
    105 [768, 768, 12] x*v
    106 [768, 768, 12] diag_mask_inf(x)
    107 [768, 768, 12] soft_max(x)
    108 [64, 768, 12] X*Y
    109 [64, 12, 768] permute(x)
    110 [768, 768] cont(x)
    111 [768, 768] X*Y
    112 [768, 768] x+y
    113 [768, 768] x+y
    114 [768, 768] norm(x)
    115 [768, 768] x*y
    116 [768, 768] x+y
    117 [3072, 768] X*Y
    118 [3072, 768] x+y
    119 [3072, 768] unary(x)
    120 [768, 768] X*Y
    121 [768, 768] x+y
    122 [768, 768] x+y
    123 [768, 768] norm(x)
    124 [768, 768] x*y
    125 [768, 768] x+y
    126 [2304, 768] X*Y
    127 [2304, 768] x+y
    128 [768, 768] view(x)
    129 [589824, 1] view(x)
    130 [589824, 1] x-\>y
    131 [768, 768] view(x)
    132 [589824, 1] view(x)
    133 [589824, 1] x-\>y
    134 [589824, 1] view(x)
    135 [64, 12, 768] reshape(x)
    136 [768, 64, 12] permute(x)
    137 [768, 64, 12] cont(x)
    138 [589824, 1] view(x)
    139 [64, 12, 768] reshape(x)
    140 [64, 768, 12] permute(x)
    141 [768, 768] view(x)
    142 [64, 12, 768] cont(x)
    143 [64, 768, 12] permute(x)
    144 [768, 768, 12] X*Y
    145 [768, 768, 12] x*v
    146 [768, 768, 12] diag_mask_inf(x)
    147 [768, 768, 12] soft_max(x)
    148 [64, 768, 12] X*Y
    149 [64, 12, 768] permute(x)
    150 [768, 768] cont(x)
    151 [768, 768] X*Y
    152 [768, 768] x+y
    153 [768, 768] x+y
    154 [768, 768] norm(x)
    155 [768, 768] x*y
    156 [768, 768] x+y
    157 [3072, 768] X*Y
    158 [3072, 768] x+y
    159 [3072, 768] unary(x)
    160 [768, 768] X*Y
    161 [768, 768] x+y
    162 [768, 768] x+y
    163 [768, 768] norm(x)
    164 [768, 768] x*y
    165 [768, 768] x+y
    166 [2304, 768] X*Y
    167 [2304, 768] x+y
    168 [768, 768] view(x)
    169 [589824, 1] view(x)
    170 [589824, 1] x-\>y
    171 [768, 768] view(x)
    172 [589824, 1] view(x)
    173 [589824, 1] x-\>y
    174 [589824, 1] view(x)
    175 [64, 12, 768] reshape(x)
    176 [768, 64, 12] permute(x)
    177 [768, 64, 12] cont(x)
    178 [589824, 1] view(x)
    179 [64, 12, 768] reshape(x)
    180 [64, 768, 12] permute(x)
    181 [768, 768] view(x)
    182 [64, 12, 768] cont(x)
    183 [64, 768, 12] permute(x)
    184 [768, 768, 12] X*Y
    185 [768, 768, 12] x*v
    186 [768, 768, 12] diag_mask_inf(x)
    187 [768, 768, 12] soft_max(x)
    188 [64, 768, 12] X*Y
    189 [64, 12, 768] permute(x)
    190 [768, 768] cont(x)
    191 [768, 768] X*Y
    192 [768, 768] x+y
    193 [768, 768] x+y
    194 [768, 768] norm(x)
    195 [768, 768] x*y
    196 [768, 768] x+y
    197 [3072, 768] X*Y
    198 [3072, 768] x+y
    199 [3072, 768] unary(x)
    200 [768, 768] X*Y
    201 [768, 768] x+y
    202 [768, 768] x+y
    203 [768, 768] norm(x)
    204 [768, 768] x*y
    205 [768, 768] x+y
    206 [2304, 768] X*Y
    207 [2304, 768] x+y
    208 [768, 768] view(x)
    209 [589824, 1] view(x)
    210 [589824, 1] x-\>y
    211 [768, 768] view(x)
    212 [589824, 1] view(x)
    213 [589824, 1] x-\>y
    214 [589824, 1] view(x)
    215 [64, 12, 768] reshape(x)
    216 [768, 64, 12] permute(x)
    217 [768, 64, 12] cont(x)
    218 [589824, 1] view(x)
    219 [64, 12, 768] reshape(x)
    220 [64, 768, 12] permute(x)
    221 [768, 768] view(x)
    222 [64, 12, 768] cont(x)
    223 [64, 768, 12] permute(x)
    224 [768, 768, 12] X*Y
    225 [768, 768, 12] x*v
    226 [768, 768, 12] diag_mask_inf(x)
    227 [768, 768, 12] soft_max(x)
    228 [64, 768, 12] X*Y
    229 [64, 12, 768] permute(x)
    230 [768, 768] cont(x)
    231 [768, 768] X*Y
    232 [768, 768] x+y
    233 [768, 768] x+y
    234 [768, 768] norm(x)
    235 [768, 768] x*y
    236 [768, 768] x+y
    237 [3072, 768] X*Y
    238 [3072, 768] x+y
    239 [3072, 768] unary(x)
    240 [768, 768] X*Y
    241 [768, 768] x+y
    242 [768, 768] x+y
    243 [768, 768] norm(x)
    244 [768, 768] x*y
    245 [768, 768] x+y
    246 [2304, 768] X*Y
    247 [2304, 768] x+y
    248 [768, 768] view(x)
    249 [589824, 1] view(x)
    250 [589824, 1] x-\>y
    251 [768, 768] view(x)
    252 [589824, 1] view(x)
    253 [589824, 1] x-\>y
    254 [589824, 1] view(x)
    255 [64, 12, 768] reshape(x)
    256 [768, 64, 12] permute(x)
    257 [768, 64, 12] cont(x)
    258 [589824, 1] view(x)
    259 [64, 12, 768] reshape(x)
    260 [64, 768, 12] permute(x)
    261 [768, 768] view(x)
    262 [64, 12, 768] cont(x)
    263 [64, 768, 12] permute(x)
    264 [768, 768, 12] X*Y
    265 [768, 768, 12] x*v
    266 [768, 768, 12] diag_mask_inf(x)
    267 [768, 768, 12] soft_max(x)
    268 [64, 768, 12] X*Y
    269 [64, 12, 768] permute(x)
    270 [768, 768] cont(x)
    271 [768, 768] X*Y
    272 [768, 768] x+y
    273 [768, 768] x+y
    274 [768, 768] norm(x)
    275 [768, 768] x*y
    276 [768, 768] x+y
    277 [3072, 768] X*Y
    278 [3072, 768] x+y
    279 [3072, 768] unary(x)
    280 [768, 768] X*Y
    281 [768, 768] x+y
    282 [768, 768] x+y
    283 [768, 768] norm(x)
    284 [768, 768] x*y
    285 [768, 768] x+y
    286 [2304, 768] X*Y
    287 [2304, 768] x+y
    288 [768, 768] view(x)
    289 [589824, 1] view(x)
    290 [589824, 1] x-\>y
    291 [768, 768] view(x)
    292 [589824, 1] view(x)
    293 [589824, 1] x-\>y
    294 [589824, 1] view(x)
    295 [64, 12, 768] reshape(x)
    296 [768, 64, 12] permute(x)
    297 [768, 64, 12] cont(x)
    298 [589824, 1] view(x)
    299 [64, 12, 768] reshape(x)
    300 [64, 768, 12] permute(x)
    301 [768, 768] view(x)
    302 [64, 12, 768] cont(x)
    303 [64, 768, 12] permute(x)
    304 [768, 768, 12] X*Y
    305 [768, 768, 12] x*v
    306 [768, 768, 12] diag_mask_inf(x)
    307 [768, 768, 12] soft_max(x)
    308 [64, 768, 12] X*Y
    309 [64, 12, 768] permute(x)
    310 [768, 768] cont(x)
    311 [768, 768] X*Y
    312 [768, 768] x+y
    313 [768, 768] x+y
    314 [768, 768] norm(x)
    315 [768, 768] x*y
    316 [768, 768] x+y
    317 [3072, 768] X*Y
    318 [3072, 768] x+y
    319 [3072, 768] unary(x)
    320 [768, 768] X*Y
    321 [768, 768] x+y
    322 [768, 768] x+y
    323 [768, 768] norm(x)
    324 [768, 768] x*y
    325 [768, 768] x+y
    326 [2304, 768] X*Y
    327 [2304, 768] x+y
    328 [768, 768] view(x)
    329 [589824, 1] view(x)
    330 [589824, 1] x-\>y
    331 [768, 768] view(x)
    332 [589824, 1] view(x)
    333 [589824, 1] x-\>y
    334 [589824, 1] view(x)
    335 [64, 12, 768] reshape(x)
    336 [768, 64, 12] permute(x)
    337 [768, 64, 12] cont(x)
    338 [589824, 1] view(x)
    339 [64, 12, 768] reshape(x)
    340 [64, 768, 12] permute(x)
    341 [768, 768] view(x)
    342 [64, 12, 768] cont(x)
    343 [64, 768, 12] permute(x)
    344 [768, 768, 12] X*Y
    345 [768, 768, 12] x*v
    346 [768, 768, 12] diag_mask_inf(x)
    347 [768, 768, 12] soft_max(x)
    348 [64, 768, 12] X*Y
    349 [64, 12, 768] permute(x)
    350 [768, 768] cont(x)
    351 [768, 768] X*Y
    352 [768, 768] x+y
    353 [768, 768] x+y
    354 [768, 768] norm(x)
    355 [768, 768] x*y
    356 [768, 768] x+y
    357 [3072, 768] X*Y
    358 [3072, 768] x+y
    359 [3072, 768] unary(x)
    360 [768, 768] X*Y
    361 [768, 768] x+y
    362 [768, 768] x+y
    363 [768, 768] norm(x)
    364 [768, 768] x*y
    365 [768, 768] x+y
    366 [2304, 768] X*Y
    367 [2304, 768] x+y
    368 [768, 768] view(x)
    369 [589824, 1] view(x)
    370 [589824, 1] x-\>y
    371 [768, 768] view(x)
    372 [589824, 1] view(x)
    373 [589824, 1] x-\>y
    374 [589824, 1] view(x)
    375 [64, 12, 768] reshape(x)
    376 [768, 64, 12] permute(x)
    377 [768, 64, 12] cont(x)
    378 [589824, 1] view(x)
    379 [64, 12, 768] reshape(x)
    380 [64, 768, 12] permute(x)
    381 [768, 768] view(x)
    382 [64, 12, 768] cont(x)
    383 [64, 768, 12] permute(x)
    384 [768, 768, 12] X*Y
    385 [768, 768, 12] x*v
    386 [768, 768, 12] diag_mask_inf(x)
    387 [768, 768, 12] soft_max(x)
    388 [64, 768, 12] X*Y
    389 [64, 12, 768] permute(x)
    390 [768, 768] cont(x)
    391 [768, 768] X*Y
    392 [768, 768] x+y
    393 [768, 768] x+y
    394 [768, 768] norm(x)
    395 [768, 768] x*y
    396 [768, 768] x+y
    397 [3072, 768] X*Y
    398 [3072, 768] x+y
    399 [3072, 768] unary(x)
    400 [768, 768] X*Y
    401 [768, 768] x+y
    402 [768, 768] x+y
    403 [768, 768] norm(x)
    404 [768, 768] x*y
    405 [768, 768] x+y
    406 [2304, 768] X*Y
    407 [2304, 768] x+y
    408 [768, 768] view(x)
    409 [589824, 1] view(x)
    410 [589824, 1] x-\>y
    411 [768, 768] view(x)
    412 [589824, 1] view(x)
    413 [589824, 1] x-\>y
    414 [589824, 1] view(x)
    415 [64, 12, 768] reshape(x)
    416 [768, 64, 12] permute(x)
    417 [768, 64, 12] cont(x)
    418 [589824, 1] view(x)
    419 [64, 12, 768] reshape(x)
    420 [64, 768, 12] permute(x)
    421 [768, 768] view(x)
    422 [64, 12, 768] cont(x)
    423 [64, 768, 12] permute(x)
    424 [768, 768, 12] X*Y
    425 [768, 768, 12] x*v
    426 [768, 768, 12] diag_mask_inf(x)
    427 [768, 768, 12] soft_max(x)
    428 [64, 768, 12] X*Y
    429 [64, 12, 768] permute(x)
    430 [768, 768] cont(x)
    431 [768, 768] X*Y
    432 [768, 768] x+y
    433 [768, 768] x+y
    434 [768, 768] norm(x)
    435 [768, 768] x*y
    436 [768, 768] x+y
    437 [3072, 768] X*Y
    438 [3072, 768] x+y
    439 [3072, 768] unary(x)
    440 [768, 768] X*Y
    441 [768, 768] x+y
    442 [768, 768] x+y
    443 [768, 768] norm(x)
    444 [768, 768] x*y
    445 [768, 768] x+y
    446 [2304, 768] X*Y
    447 [2304, 768] x+y
    448 [768, 768] view(x)
    449 [589824, 1] view(x)
    450 [589824, 1] x-\>y
    451 [768, 768] view(x)
    452 [589824, 1] view(x)
    453 [589824, 1] x-\>y
    454 [589824, 1] view(x)
    455 [64, 12, 768] reshape(x)
    456 [768, 64, 12] permute(x)
    457 [768, 64, 12] cont(x)
    458 [589824, 1] view(x)
    459 [64, 12, 768] reshape(x)
    460 [64, 768, 12] permute(x)
    461 [768, 768] view(x)
    462 [64, 12, 768] cont(x)
    463 [64, 768, 12] permute(x)
    464 [768, 768, 12] X*Y
    465 [768, 768, 12] x*v
    466 [768, 768, 12] diag_mask_inf(x)
    467 [768, 768, 12] soft_max(x)
    468 [64, 768, 12] X*Y
    469 [64, 12, 768] permute(x)
    470 [768, 768] cont(x)
    471 [768, 768] X*Y
    472 [768, 768] x+y
    473 [768, 768] x+y
    474 [768, 768] norm(x)
    475 [768, 768] x*y
    476 [768, 768] x+y
    477 [3072, 768] X*Y
    478 [3072, 768] x+y
    479 [3072, 768] unary(x)
    480 [768, 768] X*Y
    481 [768, 768] x+y
    482 [768, 768] x+y
    483 [768, 768] norm(x)
    484 [768, 768] x*y
    485 [768, 768] x+y
    486 [50257, 768] X*Y |}];

  Functions.model_uninit (addr model);
  keep model;
  ()
