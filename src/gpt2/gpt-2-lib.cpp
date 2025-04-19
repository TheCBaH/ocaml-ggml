#define main ggml_main
#include "gpt-2-lib.h"
#include "gpt-2/main-backend.cpp"

struct gpt2_model_buf {
    gpt_vocab vocab;
    gpt2_model model;
};

int gpt2_model_init(struct gpt2_model_t* model, const char* fname, int n_ctx, int n_gpu_layers)
{
    std::unique_ptr<gpt2_model_buf> buf(new gpt2_model_buf);
    std::string _fname(fname);
    if (!gpt2_model_load(_fname, buf->model, buf->vocab, n_ctx, n_gpu_layers)) {
        return -1;
    }
    model->buf = buf.release();
    return 0;
}

void gpt2_model_uninit(struct gpt2_model_t* model)
{
    delete model->buf;
}

struct ggml_cgraph * gpt2_model_graph(const struct gpt2_model_t* model, int n_past, int n_tokens)
{
    return gpt2_graph(model->buf->model, n_past, n_tokens);
}
