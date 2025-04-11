build: static.build shared.build

OPTS.static=-DBUILD_SHARED_LIBS=OFF and -DCMAKE_POSITION_INDEPENDENT_CODE=ON
TARGET.static=src/libggml.a
TARGET.shared=src/libggml.so

%.build:
	set -eux;cd ggml;\
	 cmake -B build.ml.$(basename $@) -G Ninja -DCMAKE_BUILD_TYPE=Release $(OPTS.$(basename $@));\
	 cmake --build build.ml.$(basename $@) --target $(TARGET.$(basename $@)) -j $$(getconf _NPROCESSORS_ONLN)