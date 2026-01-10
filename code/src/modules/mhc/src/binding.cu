#include <torch/extension.h>

torch::Tensor mhc_sinkhorn_cuda_forward(torch::Tensor input, int n_iters);
torch::Tensor mhc_sinkhorn_cuda_backward(torch::Tensor grad_output, torch::Tensor output, int n_iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_forward", &mhc_sinkhorn_cuda_forward, "MHC Sinkhorn Forward (CUDA)");
    m.def("sinkhorn_backward", &mhc_sinkhorn_cuda_backward, "MHC Sinkhorn Backward (CUDA)");
}