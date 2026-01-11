#include <torch/extension.h>
#include <vector>

void launch_swiglu_fwd_cuda(torch::Tensor w1, torch::Tensor w2, torch::Tensor out);
void launch_swiglu_bwd_cuda(torch::Tensor g, torch::Tensor w1, torch::Tensor w2, torch::Tensor dw1, torch::Tensor dw2);

torch::Tensor swiglu_fwd(torch::Tensor w1, torch::Tensor w2) {
    TORCH_CHECK(w1.sizes() == w2.sizes(), "w1 and w2 must have same shape");
    TORCH_CHECK(w1.is_cuda() && w2.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous(), "Inputs must be contiguous"); 

    auto out = torch::empty_like(w1);
    launch_swiglu_fwd_cuda(w1, w2, out);
    return out;
}

std::vector<torch::Tensor> swiglu_bwd(torch::Tensor grad_out, torch::Tensor w1, torch::Tensor w2) {
    TORCH_CHECK(grad_out.is_contiguous(), "Grad must be contiguous");
    
    auto dw1 = torch::empty_like(w1);
    auto dw2 = torch::empty_like(w2);
    
    launch_swiglu_bwd_cuda(grad_out, w1, w2, dw1, dw2);
    
    return {dw1, dw2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &swiglu_fwd, "SwiGLU Forward Optimized");
    m.def("bwd", &swiglu_bwd, "SwiGLU Backward Optimized");
}