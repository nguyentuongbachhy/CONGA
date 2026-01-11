#include <torch/extension.h>
#include <vector>

void launch_rope_fwd_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out);
void launch_rope_bwd_cuda(torch::Tensor gq_in, torch::Tensor gk_in, torch::Tensor cos, torch::Tensor sin, torch::Tensor gq, torch::Tensor gk);

std::vector<torch::Tensor> rope_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(q.dim() == 4, "Input must be 4D [B, L, H, D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have same shape");
    if (!q.is_contiguous()) q = q.contiguous();
    if (!k.is_contiguous()) k = k.contiguous();
    if (!cos.is_contiguous()) cos = cos.contiguous();
    if (!sin.is_contiguous()) sin = sin.contiguous();

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    
    launch_rope_fwd_cuda(q, k, cos, sin, q_out, k_out);
    
    return {q_out, k_out};
}

std::vector<torch::Tensor> rope_bwd(
    torch::Tensor grad_q_out,
    torch::Tensor grad_k_out,
    torch::Tensor cos,
    torch::Tensor sin
) {
    if (!grad_q_out.is_contiguous()) grad_q_out = grad_q_out.contiguous();
    if (!grad_k_out.is_contiguous()) grad_k_out = grad_k_out.contiguous();
    if (!cos.is_contiguous()) cos = cos.contiguous();
    if (!sin.is_contiguous()) sin = sin.contiguous();
    
    auto grad_q = torch::empty_like(grad_q_out);
    auto grad_k = torch::empty_like(grad_k_out);
    
    launch_rope_bwd_cuda(grad_q_out, grad_k_out, cos, sin, grad_q, grad_k);
    
    return {grad_q, grad_k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &rope_fwd, "RoPE forward optimized");
    m.def("bwd", &rope_bwd, "RoPE backward optimized");
}