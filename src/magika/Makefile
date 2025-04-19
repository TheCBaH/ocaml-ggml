
ROOT=${CURDIR}/../..
GGML?=${ROOT}/vendored/ggml
BIN?=${ROOT}/_build/default

CPPFLAGS=-W -Wall -L${BIN}/lib/ggml -I${BIN}/lib/ggml -I${GGML}/examples -fpic -FPIC

all: lib

lib: libmagika.a
	touch dllmagika.so

libmagika.a: magika-lib.cpp
	g++ ${CPPFLAGS} -c magika-lib.cpp -o magika-lib.o
	ar rcs $@ magika-lib.o

${BIN}/magika.bin: magika-main.cpp libmagika.a
	g++ ${CPPFLAGS} -o $@ $^ -lggml -lstdc++ -fopenmp

magika: ${BIN}/magika.bin
	cd ${GGML};$(realpath $^) ${ROOT}/test/models/magika.h5.gguf $(realpath dune $(wildcard *.ml *.h *.cpp))

clean:
	rm -vf *.so *.a *.o

.PHONY: magika clean
