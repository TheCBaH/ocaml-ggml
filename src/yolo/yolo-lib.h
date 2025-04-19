#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct yolo_model_buf;
struct yolo_model_t {
    struct yolo_model_buf* buf;
};
struct ggml_cgraph;

int ggml_main(int argc, char** argv);
int yolo_model_init(struct yolo_model_t* model, const char* fname);
void yolo_model_uninit(struct yolo_model_t* model);

struct ggml_cgraph* yolo_model_graph(const struct yolo_model_t* modelt);

#ifdef __cplusplus
}
#endif