
default:
	opam exec dune build

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
