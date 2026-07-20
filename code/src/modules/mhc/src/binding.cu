#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

constexpr int N = 4;

void launch_compute_rstd(const void* x, float* rstd, float eps, int total, int dim, cudaStream_t stream, int dtype_code);
void launch_fused_act_fwd(const void* proj_raw, const void* bias, const float* rstd, const float* a_pre, const float* a_post, const float* a_res, void* h_pre, void* h_post, void* h_res, int total, cudaStream_t stream, int dtype_code);
void launch_fused_act_bwd(const void* gh_pre, const void* gh_post, const void* gh_res, const void* proj_raw, const void* bias, const float* rstd, const void* h_pre, const void* h_post, const float* a_pre, const float* a_post, const float* a_res, void* g_proj_raw, float* g_bias, float* ga_pre, float* ga_post, float* ga_res, float* g_rstd, int total, cudaStream_t stream, int dtype_code);
void launch_apply_rstd_grad(const void* grad_x_matmul, const float* grad_rstd, const float* rstd, const void* x, void* grad_x, int total, int dim, cudaStream_t stream, int dtype_code);

std::vector<torch::Tensor> mhc_fused_forward(
    torch::Tensor x_streams,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor rms_weight,
    torch::Tensor alpha_pre,
    torch::Tensor alpha_post,
    torch::Tensor alpha_res,
    float rms_eps
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x_streams.scalar_type();
    int dtype_code = (dtype == torch::kFloat16) ? 1 : ((dtype == torch::kBFloat16) ? 2 : 0);

    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int input_dim = x_streams.size(2) * x_streams.size(3);
    int total_tokens = B * L;

    torch::Tensor x_flat = x_streams.view({-1, input_dim}).contiguous();
    torch::Tensor weight_f = proj_weight.contiguous();
    torch::Tensor bias_f = proj_bias.contiguous();
    torch::Tensor rms_w_f = rms_weight.contiguous();

    auto opts = x_flat.options();

    auto effective_weight = weight_f * rms_w_f.unsqueeze(0);
    auto proj_raw = at::matmul(x_flat, effective_weight.t());

    auto rstd = torch::empty({total_tokens}, opts.dtype(torch::kFloat32));
    launch_compute_rstd(
        x_flat.data_ptr(),
        rstd.data_ptr<float>(),
        rms_eps, total_tokens, input_dim, stream, dtype_code
    );

    if (!bias_f.defined() || bias_f.numel() == 0) {
        bias_f = torch::zeros({weight_f.size(0)}, opts);
    }

    auto H_pre = torch::empty({B, L, N}, opts);
    auto H_post = torch::empty({B, L, N}, opts);
    auto H_res = torch::empty({B, L, N, N}, opts);

    auto a_pre = alpha_pre.to(torch::kFloat32);
    auto a_post = alpha_post.to(torch::kFloat32);
    auto a_res = alpha_res.to(torch::kFloat32);

    launch_fused_act_fwd(
        proj_raw.data_ptr(),
        bias_f.data_ptr(),
        rstd.data_ptr<float>(),
        a_pre.data_ptr<float>(), a_post.data_ptr<float>(), a_res.data_ptr<float>(),
        H_pre.data_ptr(),
        H_post.data_ptr(),
        H_res.data_ptr(),
        total_tokens, stream, dtype_code
    );

    return {H_pre, H_post, H_res, proj_raw, rstd};
}

std::vector<torch::Tensor> mhc_fused_backward(
    torch::Tensor grad_H_pre,
    torch::Tensor grad_H_post,
    torch::Tensor grad_H_res,
    torch::Tensor x_streams,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor rms_weight,
    torch::Tensor H_pre,
    torch::Tensor H_post,
    torch::Tensor proj_raw,
    torch::Tensor rstd,
    torch::Tensor alpha_pre,
    torch::Tensor alpha_post,
    torch::Tensor alpha_res,
    float rms_eps
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x_streams.scalar_type();
    int dtype_code = (dtype == torch::kFloat16) ? 1 : ((dtype == torch::kBFloat16) ? 2 : 0);

    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int n = x_streams.size(2);
    int C = x_streams.size(3);
    int input_dim = n * C;
    int total_tokens = B * L;
    int out_dim = N + N + 2;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x_streams.device());

    torch::Tensor x_flat = x_streams.view({-1, input_dim}).contiguous();
    torch::Tensor weight_f = proj_weight.contiguous();
    torch::Tensor bias_f = proj_bias.contiguous();
    torch::Tensor rms_w_f = rms_weight.contiguous();

    auto grad_proj_raw = torch::empty_like(proj_raw);
    auto grad_proj_bias = torch::zeros({out_dim}, opts_f32);
    auto grad_alpha_pre = torch::zeros({1}, opts_f32);
    auto grad_alpha_post = torch::zeros({1}, opts_f32);
    auto grad_alpha_res = torch::zeros({1}, opts_f32);
    auto grad_rstd = torch::zeros({total_tokens}, opts_f32);

    torch::Tensor g_H_pre = grad_H_pre.view({-1, N}).contiguous();
    torch::Tensor g_H_post = grad_H_post.view({-1, N}).contiguous();
    torch::Tensor g_H_res = grad_H_res.view({-1, N * N}).contiguous();
    torch::Tensor H_pre_f = H_pre.view({-1, N}).contiguous();
    torch::Tensor H_post_f = H_post.view({-1, N}).contiguous();

    auto a_pre = alpha_pre.to(torch::kFloat32);
    auto a_post = alpha_post.to(torch::kFloat32);
    auto a_res = alpha_res.to(torch::kFloat32);

    launch_fused_act_bwd(
        g_H_pre.data_ptr(),
        g_H_post.data_ptr(),
        g_H_res.data_ptr(),
        proj_raw.data_ptr(),
        bias_f.data_ptr(),
        rstd.data_ptr<float>(),
        H_pre_f.data_ptr(),
        H_post_f.data_ptr(),
        a_pre.data_ptr<float>(), a_post.data_ptr<float>(), a_res.data_ptr<float>(),
        grad_proj_raw.data_ptr(),
        grad_proj_bias.data_ptr<float>(),
        grad_alpha_pre.data_ptr<float>(),
        grad_alpha_post.data_ptr<float>(),
        grad_alpha_res.data_ptr<float>(),
        grad_rstd.data_ptr<float>(),
        total_tokens, stream, dtype_code
    );

    auto effective_weight = weight_f * rms_w_f.unsqueeze(0);

    torch::Tensor eff_w_for_grad = effective_weight;
    torch::Tensor x_flat_for_grad = x_flat;
    torch::Tensor grad_proj_raw_f = grad_proj_raw;

    if (effective_weight.scalar_type() != grad_proj_raw.scalar_type())
        eff_w_for_grad = effective_weight.to(grad_proj_raw.scalar_type());
    if (x_flat.scalar_type() != grad_proj_raw.scalar_type())
        x_flat_for_grad = x_flat.to(grad_proj_raw.scalar_type());

    auto grad_x_matmul = at::matmul(grad_proj_raw_f, eff_w_for_grad);
    auto grad_effective_weight = at::matmul(grad_proj_raw_f.t(), x_flat_for_grad);

    torch::Tensor grad_x_matmul_cast = grad_x_matmul;
    if (grad_x_matmul.scalar_type() != x_flat.scalar_type())
        grad_x_matmul_cast = grad_x_matmul.to(x_flat.scalar_type());

    auto grad_x = torch::empty_like(x_flat);
    launch_apply_rstd_grad(
        grad_x_matmul_cast.data_ptr(),
        grad_rstd.data_ptr<float>(),
        rstd.data_ptr<float>(),
        x_flat.data_ptr(),
        grad_x.data_ptr(),
        total_tokens, input_dim, stream, dtype_code
    );

    torch::Tensor grad_proj_weight, grad_rms_weight;
    if (grad_effective_weight.scalar_type() != rms_w_f.scalar_type()) {
        auto grad_eff_w_cast = grad_effective_weight.to(rms_w_f.scalar_type());
        grad_proj_weight = grad_eff_w_cast * rms_w_f.unsqueeze(0);
        grad_rms_weight = (grad_eff_w_cast * weight_f).sum(0);
    } else {
        grad_proj_weight = grad_effective_weight * rms_w_f.unsqueeze(0);
        grad_rms_weight = (grad_effective_weight * weight_f).sum(0);
    }

    torch::Tensor grad_x_out = grad_x.view({B, L, n, C});

    torch::Tensor grad_proj_bias_out = grad_proj_bias;
    if (dtype != torch::kFloat32)
        grad_proj_bias_out = grad_proj_bias.to(dtype);

    if (grad_proj_weight.scalar_type() != proj_weight.scalar_type())
        grad_proj_weight = grad_proj_weight.to(proj_weight.scalar_type());

    return {
        grad_x_out,
        grad_proj_weight,
        grad_proj_bias_out,
        grad_rms_weight,
        grad_alpha_pre,
        grad_alpha_post,
        grad_alpha_res
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward",  &mhc_fused_forward,  "MHC KromHC Fused Forward (CUDA)");
    m.def("fused_backward", &mhc_fused_backward, "MHC KromHC Fused Backward (CUDA)");
}
