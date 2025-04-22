build-ggml: static.build-ggml shared.build-ggml
build-llama: static.build-llama shared.build-llama

OPTS.static=-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
TARGET.GGML.static=ggml/src/libggml.a
TARGET.GGML.shared=bin/libggml.so
FINISH.GGML.static=printf "create libggml.a\naddlib build.ml.static/ggml/src/libggml.a\n addlib build.ml.static/ggml/src/libggml-base.a\n addlib build.ml.static/ggml/src/libggml-cpu.a\naddlib build.ml.static/ggml/src/libggml-cpu.a\nsave\nend" | ar -M;
ROOT=${CURDIR}/../..
LLAMA.CPP?=${ROOT}/vendored/llama.cpp
TARGET.LLAMA.CPP.static=src/libllama.a
TARGET.LLAMA.CPP.shared=bin/libllama.so
BIN_DIR=.

%.config:
	set -eux;cd ${LLAMA.CPP};\
	 cmake -B ${BIN_DIR}/build.ml.$(basename $@) -G Ninja -DCMAKE_BUILD_TYPE=Release $(OPTS.$(basename $@))

%.build-ggml: %.config
	set -eux;cd ${LLAMA.CPP};\
	 cmake --build ${BIN_DIR}/build.ml.$(basename $@) --target $(TARGET.GGML.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN);\
	 $(FINISH.GGML.$(basename $@))\
	 true

%.build-llama: %.config
	set -eux;cd ${LLAMA.CPP};\
	 cmake --build ${BIN_DIR}/build.ml.$(basename $@) --target $(TARGET.LLAMA.CPP.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN);\
	 true

TARGET.LLAMA.CPP.simple=bin/llama-simple
simple: BIN_DIR=$(abspath ${ROOT})/_build/.ml.simple
simple:
	${MAKE} simple.build-llama BIN_DIR=${BIN_DIR}
	${BIN_DIR}/build.ml.simple/${TARGET.LLAMA.CPP.simple} -m ${ROOT}/bartowski/SmolLM2-135M-Instruct-Q5_K_M.gguf
