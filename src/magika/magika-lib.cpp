#define main ggml_main
#include "magika-lib.h"
#include "magika/main.cpp"
#include <memory>

struct magika_model_buf {
    magika_model model;
};

int magika_model_init(struct magika_model_t* model, const char* fname)
{
    std::unique_ptr<magika_model_buf> buf(new magika_model_buf);
    std::string _fname(fname);
    if (!magika_model_load(_fname, buf->model)) {
        return -1;
    }
    model->buf = buf.release();
    return 0;
}

void magika_model_uninit(struct magika_model_t* model)
{
    delete model->buf;
}

struct ggml_cgraph * magika_model_graph(const struct magika_model_t* model)
{
    return magika_graph(model->buf->model, 1);
}
