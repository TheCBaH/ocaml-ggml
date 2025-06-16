#define main yolo_main
#include "yolo-lib.h"
#include "yolo/yolov3-tiny.cpp"
#include <memory>

struct yolo_model_buf {
    struct ggml_context* ctx;
    yolo_model model;
};

int yolo_model_init(struct yolo_model_t* model, const char* fname)
{
    std::unique_ptr<yolo_model_buf> buf(new yolo_model_buf);
    std::string _fname(fname);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true
    };
    struct yolo_params yolo_params;
    buf->model.backend = create_backend(yolo_params);
    if (!buf->model.backend) {
        return -1;
    }
    if (!load_model(_fname, buf->model)) {
        return -1;
    }
    buf->ctx = ggml_init(params);
    model->buf = buf.release();
    return 0;
}

void yolo_model_uninit(struct yolo_model_t* model)
{
    ggml_free(model->buf->ctx);
    delete model->buf;
}

struct ggml_cgraph * yolo_model_graph(const struct yolo_model_t* model)
{
    return build_graph(model->buf->ctx, model->buf->model);
}
