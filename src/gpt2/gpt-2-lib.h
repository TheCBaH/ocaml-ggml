#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gpt2_model_buf;
struct gpt2_model_t {
    struct gpt2_model_buf* buf;
};
struct ggml_cgraph;

int gpt2_main(int argc, char ** argv);
int gpt2_model_init(struct gpt2_model_t* model, const char* fname, int n_ctx, int n_gpu_layers);
void gpt2_model_uninit(struct gpt2_model_t* model);

struct ggml_cgraph * gpt2_model_graph(const struct gpt2_model_t* model, int n_past, int n_tokens);

#ifdef __cplusplus
}
#endif