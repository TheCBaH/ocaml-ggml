build: static.build shared.build

OPTS.static=-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
TARGET.static=src/libggml.a
TARGET.shared=src/libggml.so
FINISH.static=printf "create libggml.a\naddlib build.ml.static/src/libggml.a\n addlib build.ml.static/src/libggml-base.a\naddlib build.ml.static/src/libggml-cpu.a\nsave\nend" | ar -M;

%.build:
	set -eux;cd ${GGML};\
	 cmake -B build.ml.$(basename $@) -G Ninja -DCMAKE_BUILD_TYPE=Release $(OPTS.$(basename $@));\
	 cmake --build build.ml.$(basename $@) --target $(TARGET.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN);\
	 $(FINISH.$(basename $@))\
	 true