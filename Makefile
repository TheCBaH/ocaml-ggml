
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

.PHONY: default clean format run top utop

server.model-explorer.js:
	cd model-explorer/src/custom_element_demos/vanilla_js; ./build_and_deploy.sh

server.model-explorer.ts:
	cd model-explorer/src/custom_element_demos/vanilla_ts; npm run build_and_deploy

server.stop:
	${CURDIR}/scripts/server.sh stop
