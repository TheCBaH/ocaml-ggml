
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

models:  test/models/gpt-2-117M/ggml-model.bin

.PHONY: default clean format models run top utop
