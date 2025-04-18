open Ctypes
open Gpt_2.C
open Model_explorer

let keep x = ignore (Sys.opaque_identity (List.hd [ x ]))
let getfp p field = !@(p |-> field)
let to_string t = Ctypes.(coerce (ptr char) string t)

let attr key value = KeyValue.create ~key ~value

(*
let tensor t =
  let name = getfp t Ggml.C.Types.Tensor.name in
  let name = to_string @@ CArray.start name in
  let tensor_name = attr "tensor_name" name in
  let tensor_index = attr "tensor_index" "0" in
  let tensor_shape = if Ggml.C.Functions.is_matrix
*)

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
  [%expect {| nodes:487 |}];

  let nodes = Array.init nodes (fun n -> Ggml.C.Functions.graph_node gpt2 n) in
  let names = Array.map (fun t -> Ggml.C.Functions.op_name @@ getfp t Ggml.C.Types.Tensor.op) nodes in
  ignore names;
  let names = Array.map tensor nodes in

  Format.printf "@[%a@]" (Format.pp_print_list ~pp_sep:Format.pp_print_newline Format.pp_print_string)
  @@ Array.to_list names;
  [%expect
    {|
    node_0
    node_1
    node_2
    node_3
    node_4
    node_5
    node_6
    node_7
     (view)
     (view)
     (view) (copy of  (view))
     (view)
     (view)
     (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_24
    node_25
    node_26
    node_27
    node_28
     (permuted)
     (permuted) (cont)
    node_31
    node_32
    node_33
    node_34
    node_35
    node_36
    node_37
    node_38
    node_39
    node_40
    node_41
    node_42
    node_43
    node_44
    node_45
    node_46
    node_47
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_64
    node_65
    node_66
    node_67
    node_68
     (permuted)
     (permuted) (cont)
    node_71
    node_72
    node_73
    node_74
    node_75
    node_76
    node_77
    node_78
    node_79
    node_80
    node_81
    node_82
    node_83
    node_84
    node_85
    node_86
    node_87
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_104
    node_105
    node_106
    node_107
    node_108
     (permuted)
     (permuted) (cont)
    node_111
    node_112
    node_113
    node_114
    node_115
    node_116
    node_117
    node_118
    node_119
    node_120
    node_121
    node_122
    node_123
    node_124
    node_125
    node_126
    node_127
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_144
    node_145
    node_146
    node_147
    node_148
     (permuted)
     (permuted) (cont)
    node_151
    node_152
    node_153
    node_154
    node_155
    node_156
    node_157
    node_158
    node_159
    node_160
    node_161
    node_162
    node_163
    node_164
    node_165
    node_166
    node_167
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_184
    node_185
    node_186
    node_187
    node_188
     (permuted)
     (permuted) (cont)
    node_191
    node_192
    node_193
    node_194
    node_195
    node_196
    node_197
    node_198
    node_199
    node_200
    node_201
    node_202
    node_203
    node_204
    node_205
    node_206
    node_207
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_224
    node_225
    node_226
    node_227
    node_228
     (permuted)
     (permuted) (cont)
    node_231
    node_232
    node_233
    node_234
    node_235
    node_236
    node_237
    node_238
    node_239
    node_240
    node_241
    node_242
    node_243
    node_244
    node_245
    node_246
    node_247
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_264
    node_265
    node_266
    node_267
    node_268
     (permuted)
     (permuted) (cont)
    node_271
    node_272
    node_273
    node_274
    node_275
    node_276
    node_277
    node_278
    node_279
    node_280
    node_281
    node_282
    node_283
    node_284
    node_285
    node_286
    node_287
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_304
    node_305
    node_306
    node_307
    node_308
     (permuted)
     (permuted) (cont)
    node_311
    node_312
    node_313
    node_314
    node_315
    node_316
    node_317
    node_318
    node_319
    node_320
    node_321
    node_322
    node_323
    node_324
    node_325
    node_326
    node_327
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_344
    node_345
    node_346
    node_347
    node_348
     (permuted)
     (permuted) (cont)
    node_351
    node_352
    node_353
    node_354
    node_355
    node_356
    node_357
    node_358
    node_359
    node_360
    node_361
    node_362
    node_363
    node_364
    node_365
    node_366
    node_367
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_384
    node_385
    node_386
    node_387
    node_388
     (permuted)
     (permuted) (cont)
    node_391
    node_392
    node_393
    node_394
    node_395
    node_396
    node_397
    node_398
    node_399
    node_400
    node_401
    node_402
    node_403
    node_404
    node_405
    node_406
    node_407
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_424
    node_425
    node_426
    node_427
    node_428
     (permuted)
     (permuted) (cont)
    node_431
    node_432
    node_433
    node_434
    node_435
    node_436
    node_437
    node_438
    node_439
    node_440
    node_441
    node_442
    node_443
    node_444
    node_445
    node_446
    node_447
     (view)
    leaf_8 (view)
    leaf_8 (view) (copy of  (view))
     (view)
    leaf_9 (view)
    leaf_9 (view) (copy of  (view))
    leaf_9 (view)
    leaf_9 (view) (reshaped)
    leaf_9 (view) (reshaped) (permuted)
    leaf_9 (view) (reshaped) (permuted) (cont)
    leaf_8 (view)
    leaf_8 (view) (reshaped)
    leaf_8 (view) (reshaped) (permuted)
     (view)
     (view) (cont)
     (view) (cont) (permuted)
    node_464
    node_465
    node_466
    node_467
    node_468
     (permuted)
     (permuted) (cont)
    node_471
    node_472
    node_473
    node_474
    node_475
    node_476
    node_477
    node_478
    node_479
    node_480
    node_481
    node_482
    node_483
    node_484
    node_485
    logits |}];

  Functions.model_uninit (addr model);
  keep model;
  ()
