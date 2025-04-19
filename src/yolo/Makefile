
ROOT=${CURDIR}/../..
GGML?=${ROOT}/vendored/ggml
BIN?=${ROOT}/_build/default

CPPFLAGS=-W -Wall -L${BIN}/lib/ggml -I${BIN}/lib/ggml -I${GGML}/examples -fpic -FPIC

all: lib

lib: libyolo.a
	touch dllyolo.so

libyolo.a: yolo-lib.cpp ${GGML}/examples/yolo/yolo-image.cpp
	g++ ${CPPFLAGS} -c ${GGML}/examples/yolo/yolo-image.cpp -o yolo-image.o
	g++ ${CPPFLAGS} -c yolo-lib.cpp -o yolo-lib.o
	ar rcs $@ yolo-lib.o yolo-image.o

${BIN}/yolov3-tiny.bin: yolo-main.cpp libyolo.a
	g++ ${CPPFLAGS} -o $@ $^ -lggml -lstdc++ -fopenmp

yolov3-tiny: ${BIN}/yolov3-tiny.bin
	cd ${GGML};$(realpath $^)

clean:
	rm -vf *.so *.a *.o

.PHONY: yolov3-tiny clean
