#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

void launch_rms_fwd(const float* x, const float* w, float* xn, float* rstd, float eps, int total, int dim, cudaStream_t stream);
void launch_rms_bwd(const float* dout, const float* x, const float* rstd, const float* w, float* dx, int total, int dim, cudaStream_t stream);
void launch_fused_act_fwd(const float* proj, float a_pre, float a_post, float a_res, int iters, float* h_pre, float* h_post, float* h_res, int total, cudaStream_t stream);
void launch_fused_act_bwd(const float* gh_pre, const float* gh_post, const float* gh_res, const float* proj, const float* h_pre, const float* h_post, float a_pre, float a_post, float a_res, int iters, float* g_proj, float* ga_pre, float* ga_post, float* ga_res, int total, cudaStream_t stream);

std::vector<torch::Tensor> mhc_fused_forward(
    torch::Tensor x_streams,
    torch::Tensor proj_weight,
    torch::Tensor proj_bias,
    torch::Tensor rms_weight,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    float rms_eps,
    int n_iters
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto orig_dtype = x_streams.scalar_type();
    
    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int input_dim = x_streams.size(2) * x_streams.size(3);
    int total_tokens = B * L;
    constexpr int N = 4;

    torch::Tensor x_flat, weight_f, bias_f, rms_w_f;
    
    if (orig_dtype == torch::kFloat32) {
        x_flat = x_streams.view({-1, input_dim}).contiguous();
        weight_f = proj_weight.contiguous();
        bias_f = proj_bias.contiguous();
        rms_w_f = rms_weight.contiguous();
    } else {
        x_flat = x_streams.to(torch::kFloat32).view({-1, input_dim}).contiguous();
        weight_f = proj_weight.to(torch::kFloat32).contiguous();
        bias_f = proj_bias.to(torch::kFloat32).contiguous();
        rms_w_f = rms_weight.to(torch::kFloat32).contiguous();
    }

    auto opts = x_flat.options();
    auto x_norm = torch::empty_like(x_flat);
    auto rstd = torch::empty({total_tokens}, opts);
    
    launch_rms_fwd(
        x_flat.data_ptr<float>(), rms_w_f.data_ptr<float>(),
        x_norm.data_ptr<float>(), rstd.data_ptr<float>(),
        rms_eps, total_tokens, input_dim, stream
    );

    auto proj_output = at::linear(x_norm, weight_f, bias_f);

    auto H_pre = torch::empty({B, L, N}, opts);
    auto H_post = torch::empty({B, L, N}, opts);
    auto H_res = torch::empty({B, L, N, N}, opts);

    launch_fused_act_fwd(
        proj_output.data_ptr<float>(), 
        alpha_pre, alpha_post, alpha_res, n_iters,
        H_pre.data_ptr<float>(), H_post.data_ptr<float>(), H_res.data_ptr<float>(),
        total_tokens, stream
    );

    if (orig_dtype != torch::kFloat32) {
        H_pre = H_pre.to(orig_dtype);
        H_post = H_post.to(orig_dtype);
        H_res = H_res.to(orig_dtype);
    }

    return {H_pre, H_post, H_res, x_norm, proj_output, rstd};
}

std::vector<torch::Tensor> mhc_fused_backward(
    torch::Tensor grad_H_pre,
    torch::Tensor grad_H_post,
    torch::Tensor grad_H_res,
    torch::Tensor x_norm,
    torch::Tensor x_streams,
    torch::Tensor proj_weight,
    torch::Tensor rms_weight,
    torch::Tensor H_pre,
    torch::Tensor H_post,
    torch::Tensor proj_output,
    torch::Tensor rstd,
    float alpha_pre,
    float alpha_post,
    float alpha_res,
    float rms_eps,
    int n_iters
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto orig_dtype = x_streams.scalar_type();
    
    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int n = x_streams.size(2);
    int C = x_streams.size(3);
    int input_dim = n * C;
    int total_tokens = B * L;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x_streams.device());
    auto grad_proj_output = torch::empty_like(proj_output);
    auto grad_alpha_pre = torch::zeros({1}, opts);
    auto grad_alpha_post = torch::zeros({1}, opts);
    auto grad_alpha_res = torch::zeros({1}, opts);

    torch::Tensor g_H_pre, g_H_post, g_H_res, H_pre_f, H_post_f, x_flat, weight_f, rms_w_f;
    
    if (orig_dtype == torch::kFloat32) {
        g_H_pre = grad_H_pre.view({-1, 4}).contiguous();
        g_H_post = grad_H_post.view({-1, 4}).contiguous();
        g_H_res = grad_H_res.view({-1, 16}).contiguous();
        H_pre_f = H_pre.view({-1, 4}).contiguous();
        H_post_f = H_post.view({-1, 4}).contiguous();
        x_flat = x_streams.view({-1, input_dim}).contiguous();
        weight_f = proj_weight.contiguous();
        rms_w_f = rms_weight.contiguous();
    } else {
        g_H_pre = grad_H_pre.to(torch::kFloat32).view({-1, 4}).contiguous();
        g_H_post = grad_H_post.to(torch::kFloat32).view({-1, 4}).contiguous();
        g_H_res = grad_H_res.to(torch::kFloat32).view({-1, 16}).contiguous();
        H_pre_f = H_pre.to(torch::kFloat32).view({-1, 4}).contiguous();
        H_post_f = H_post.to(torch::kFloat32).view({-1, 4}).contiguous();
        x_flat = x_streams.to(torch::kFloat32).view({-1, input_dim}).contiguous();
        weight_f = proj_weight.to(torch::kFloat32).contiguous();
        rms_w_f = rms_weight.to(torch::kFloat32).contiguous();
    }

    launch_fused_act_bwd(
        g_H_pre.data_ptr<float>(),
        g_H_post.data_ptr<float>(),
        g_H_res.data_ptr<float>(),
        proj_output.data_ptr<float>(),
        H_pre_f.data_ptr<float>(),
        H_post_f.data_ptr<float>(),
        alpha_pre, alpha_post, alpha_res, n_iters,
        grad_proj_output.data_ptr<float>(),
        grad_alpha_pre.data_ptr<float>(),
        grad_alpha_post.data_ptr<float>(),
        grad_alpha_res.data_ptr<float>(),
        total_tokens, stream
    );

    auto x_norm_flat = x_norm.view({-1, input_dim});
    auto grad_proj_weight = at::matmul(grad_proj_output.t(), x_norm_flat);
    auto grad_proj_bias = grad_proj_output.sum(0);
    auto grad_x_norm = at::matmul(grad_proj_output, weight_f);

    auto grad_x = torch::empty_like(x_flat);
    auto grad_rms_weight = (grad_x_norm * x_flat * rstd.unsqueeze(1)).sum(0);
    
    launch_rms_bwd(
        grad_x_norm.data_ptr<float>(),
        x_flat.data_ptr<float>(),
        rstd.data_ptr<float>(),
        rms_w_f.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        total_tokens, input_dim, stream
    );

    torch::Tensor grad_x_out = grad_x.view({B, L, n, C});
    if (orig_dtype != torch::kFloat32) {
        grad_x_out = grad_x_out.to(orig_dtype);
        grad_proj_weight = grad_proj_weight.to(orig_dtype);
        grad_proj_bias = grad_proj_bias.to(orig_dtype);
        grad_rms_weight = grad_rms_weight.to(orig_dtype);
    }

    return {
        grad_x_out,
        grad_proj_weight,
        grad_proj_bias,
        grad_rms_weight,
        grad_alpha_pre,
        grad_alpha_post,
        grad_alpha_res
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &mhc_fused_forward, "MHC Fused Forward (CUDA)");
    m.def("fused_backward", &mhc_fused_backward, "MHC Fused Backward (CUDA)");
}