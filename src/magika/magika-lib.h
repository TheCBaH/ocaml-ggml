#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct magika_model_buf;
struct magika_model_t {
    struct magika_model_buf* buf;
};
struct ggml_cgraph;

int ggml_main(int argc, const char** argv);
int magika_model_init(struct magika_model_t* model, const char* fname);
void magika_model_uninit(struct magika_model_t* model);

struct ggml_cgraph* magika_model_graph(const struct magika_model_t* modelt);

#ifdef __cplusplus
}
#endif