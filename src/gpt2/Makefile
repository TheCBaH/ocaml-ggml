
ROOT=${CURDIR}/../..
GGML?=${ROOT}/vendored/ggml
BIN?=${ROOT}/_build/default

CPPFLAGS=-W -Wall -L${BIN}/lib -I${BIN}/lib -I${GGML}/examples -fpic -FPIC

all: lib

lib: libgpt_2.a
	touch dllgpt_2.so

libgpt_2.a: gpt-2-lib.cpp ${GGML}/examples/common.cpp
	g++ ${CPPFLAGS} -c ${GGML}/examples/common.cpp -o common.o
	g++ ${CPPFLAGS} -c gpt-2-lib.cpp -o gpt-2-lib.o
	ar rcs $@ common.o gpt-2-lib.o

${BIN}/gpt-2-backend.bin: gpt-2-main.cpp libgpt_2.a
	g++ ${CPPFLAGS} -o $@ $^ -lggml -lstdc++ -fopenmp

gpt-2-backend: ${BIN}/gpt-2-backend.bin
	cd ${GGML};$(realpath $^)

.PHONY: gpt-2-backend
