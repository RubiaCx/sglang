#include <torch/extension.h>

#include "rotary_embedding.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &rotary_embedding, "Rotary Embedding");
}

