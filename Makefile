
default: build

build:
	opam exec -- dune $@

runtest:
	opam exec -- dune $@ --auto-promote

static:
	opam exec -- dune build --profile static

format:
	opam exec dune fmt

run:
	opam exec dune exec ./main.exe

top:
	opam exec dune exec ./example_top.exe

utop:
	opam exec dune utop

clean:
	opam exec dune $@

test/models/gpt-2-117M/ggml-model.bin:
	cd test;../vendored/ggml/examples/gpt-2/download-ggml-model.sh 117M

test/models/yolov3-tiny.gguf:
	wget -O $@ https://huggingface.co/rgerganov/yolo-gguf/resolve/main/yolov3-tiny.gguf

test/models/magika.h5.gguf:
	wget -O $(basename $@) https://github.com/google/magika/raw/4460acb5d3f86807c3b53223229dee2afa50c025/assets_generation/models/standard_v1/model.h5
	env TF_USE_LEGACY_KERAS=1 python3 vendored/ggml/examples/magika/convert.py $(basename $@)

models:  test/models/gpt-2-117M/ggml-model.bin test/models/yolov3-tiny.gguf test/models/magika.h5.gguf

model-explorer.install:
	python3 -m pip --no-cache-dir install ai-edge-model-explorer

visualize:
	opam exec -- dune exec -- ./visualize.exe magika test/models/magika.h5.gguf magika.json
	opam exec -- dune exec -- ./visualize.exe yolo test/models/yolov3-tiny.gguf yolov3-tiny.json
	opam exec -- dune exec -- ./visualize.exe gpt2 test/models/gpt-2-117M/ggml-model.bin gpt2.json

.PHONY: default clean format models run top utop model-explorer.install
