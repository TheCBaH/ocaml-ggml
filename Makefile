
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

server.image-explorer.js:
	cd src/custom_element_demos/vanilla_js; ${CURDIR}/scripts/server.sh ./build_and_deploy.sh

server.image-explorer.ts:
	cd src/custom_element_demos/vanilla_ts; npm run build_and_deploy

server.stop:
	${CURDIR}/scripts/server.sh stop
