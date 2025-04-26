
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

bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q5_K_M.gguf

models: bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf
simple: models
	mkdir -p _build/json
	opam exec -- dune exec -- src/simple.exe -g _build/json bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf

model-explorer.install:
	python3 -m pip --no-cache-dir install ai-edge-model-explorer

patch:
	git -C vendored/llama.cpp apply < patches/graph_callback_llama.patch

.PHONY: default clean format models patch run top utop model-explorer.install
