
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

models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q5_K_M.gguf

models/ggml-org/gemma-3-1b-it-Q4_K_M.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

models/ggml-org/qwen2.5-coder-0.5b-q8_0.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF/resolve/main/qwen2.5-coder-0.5b-q8_0.gguf

models/tiiuae/Falcon3-1B-Instruct-q3_k_m.gguf:
	mkdir -p $(dir $@)
	wget -O $@ https://huggingface.co/tiiuae/Falcon3-1B-Instruct-GGUF/resolve/main/Falcon3-1B-Instruct-q3_k_m.gguf

MODEL=models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf
models: models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf ${MODEL}
simple: models
	mkdir -p _build/json
	opam exec -- dune exec -- src/simple.exe -g _build/json ${MODEL}

simple.all:
	$(foreach m,\
 models/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf\
 models/ggml-org/gemma-3-1b-it-Q4_K_M.gguf\
 models/ggml-org/qwen2.5-coder-0.5b-q8_0.gguf\
 models/tiiuae/Falcon3-1B-Instruct-q3_k_m.gguf\
  , ${MAKE} MODEL=$m simple &&) true

llama-simple: ${MODEL}
	${MAKE} -C lib/llama.cpp simple MODEL=$(abspath ${MODEL})

model-explorer.install:
	python3 -m pip --no-cache-dir install ai-edge-model-explorer

patch:
	git -C vendored/llama.cpp apply < patches/graph_callback_llama.patch

.PHONY: default clean format models patch run top utop model-explorer.install
