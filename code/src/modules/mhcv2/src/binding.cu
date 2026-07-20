/*
 * MHCv2 CUDA Binding — PyTorch ↔ CUDA interface
 *
 * Exposes mhcv2_fused_forward() and mhcv2_fused_backward() to Python
 * via pybind11.  These are called from MHCv2Function (autograd Function)
 * in the Python wrapper.
 */

#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// Forward declarations from kernel.cu
void launch_mhcv2_fwd(
    const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs,
    void* out, float* save_mean, float* save_rstd,
    int total_tokens, int C, int K, int N_STREAMS,
    cudaStream_t stream, int dtype_code
);

void launch_mhcv2_bwd(
    const void* grad_out, const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs, const float* save_mean, const float* save_rstd,
    void* grad_x, void* grad_sublayer,
    float* grad_post_scale, float* grad_lw_work, float* grad_lb_work,
    float* grad_kron_probs,
    int total_tokens, int C, int K, int N_STREAMS,
    cudaStream_t stream, int dtype_code
);

static int get_dtype_code(torch::ScalarType dtype) {
    if (dtype == torch::kFloat16)   return 1;
    if (dtype == torch::kBFloat16)  return 2;
    return 0; // float32
}

// ─────────────────────── FORWARD ───────────────────────

std::vector<torch::Tensor> mhcv2_fused_forward(
    torch::Tensor x_streams,    // (B, L, N, C)
    torch::Tensor sublayer_out, // (B, L, C)
    torch::Tensor post_scale,   // (N,)
    torch::Tensor ln_weight,    // (N, C)
    torch::Tensor ln_bias,      // (N, C)
    torch::Tensor kron_probs    // (K,)  — sigmoid(kron_logits) computed by caller
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x_streams.scalar_type();
    int dtype_code = get_dtype_code(dtype);

    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int N = x_streams.size(2);
    int C = x_streams.size(3);
    int T = B * L;
    int K = kron_probs.size(0);

    auto x_flat = x_streams.reshape({T, N, C}).contiguous();
    // Only cast sublayer_out if dtype differs (avoids copy under autocast when already bf16)
    auto s_flat = (sublayer_out.scalar_type() == dtype)
                  ? sublayer_out.reshape({T, C}).contiguous()
                  : sublayer_out.to(dtype).reshape({T, C}).contiguous();

    // Parameters are always float32; only call contiguous if needed (no-op for contiguous)
    auto ps_f = post_scale.is_contiguous() ? post_scale : post_scale.contiguous();
    auto lw_f = ln_weight.is_contiguous() ? ln_weight : ln_weight.contiguous();
    auto lb_f = ln_bias.is_contiguous() ? ln_bias : ln_bias.contiguous();
    auto kp_f = kron_probs.is_contiguous() ? kron_probs : kron_probs.contiguous();

    // Output tensor
    auto out = torch::empty_like(x_flat);

    // Saved tensors for backward
    auto save_mean = torch::empty({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(x_streams.device()));
    auto save_rstd = torch::empty({T, N}, torch::TensorOptions().dtype(torch::kFloat32).device(x_streams.device()));

    launch_mhcv2_fwd(
        x_flat.data_ptr(),
        s_flat.data_ptr(),
        ps_f.data_ptr<float>(),
        lw_f.data_ptr<float>(),
        lb_f.data_ptr<float>(),
        kp_f.data_ptr<float>(),
        out.data_ptr(),
        save_mean.data_ptr<float>(),
        save_rstd.data_ptr<float>(),
        T, C, K, N,
        stream, dtype_code
    );

    // Reshape output back to (B, L, N, C)
    out = out.view({B, L, N, C});

    return {out, save_mean, save_rstd};
}


// ─────────────────────── BACKWARD ───────────────────────

std::vector<torch::Tensor> mhcv2_fused_backward(
    torch::Tensor grad_out,      // (B, L, N, C)
    torch::Tensor x_streams,     // (B, L, N, C)
    torch::Tensor sublayer_out,  // (B, L, C)
    torch::Tensor post_scale,    // (N,)
    torch::Tensor ln_weight,     // (N, C)
    torch::Tensor ln_bias,       // (N, C)
    torch::Tensor kron_probs,    // (K,)
    torch::Tensor save_mean,     // (T, N)
    torch::Tensor save_rstd      // (T, N)
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x_streams.scalar_type();
    int dtype_code = get_dtype_code(dtype);

    int B = x_streams.size(0);
    int L = x_streams.size(1);
    int N = x_streams.size(2);
    int C = x_streams.size(3);
    int T = B * L;
    int K = kron_probs.size(0);

    auto go_flat = (grad_out.scalar_type() == dtype)
                   ? grad_out.reshape({T, N, C}).contiguous()
                   : grad_out.to(dtype).reshape({T, N, C}).contiguous();
    auto x_flat  = x_streams.reshape({T, N, C}).contiguous();
    auto s_flat  = (sublayer_out.scalar_type() == dtype)
                   ? sublayer_out.reshape({T, C}).contiguous()
                   : sublayer_out.to(dtype).reshape({T, C}).contiguous();

    auto ps_f = post_scale.is_contiguous() ? post_scale : post_scale.contiguous();
    auto lw_f = ln_weight.is_contiguous() ? ln_weight : ln_weight.contiguous();
    auto lb_f = ln_bias.is_contiguous() ? ln_bias : ln_bias.contiguous();
    auto kp_f = kron_probs.is_contiguous() ? kron_probs : kron_probs.contiguous();
    auto sm_f = save_mean.contiguous();
    auto sr_f = save_rstd.contiguous();

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x_streams.device());

    // Output gradient tensors
    auto grad_x        = torch::empty_like(x_flat);
    auto grad_sublayer = torch::empty({T, C}, x_streams.options());
    auto grad_ps       = torch::zeros({N}, opts_f32);
    // Workspace for ln_weight/bias grads: each token writes its own (N, C) slot,
    // eliminating atomicAdd contention across T blocks. PyTorch sums after.
    auto grad_lw_work  = torch::empty({T, N, C}, opts_f32);
    auto grad_lb_work  = torch::empty({T, N, C}, opts_f32);
    auto grad_kp       = torch::zeros({K}, opts_f32);

    launch_mhcv2_bwd(
        go_flat.data_ptr(),
        x_flat.data_ptr(),
        s_flat.data_ptr(),
        ps_f.data_ptr<float>(),
        lw_f.data_ptr<float>(),
        lb_f.data_ptr<float>(),
        kp_f.data_ptr<float>(),
        sm_f.data_ptr<float>(),
        sr_f.data_ptr<float>(),
        grad_x.data_ptr(),
        grad_sublayer.data_ptr(),
        grad_ps.data_ptr<float>(),
        grad_lw_work.data_ptr<float>(),
        grad_lb_work.data_ptr<float>(),
        grad_kp.data_ptr<float>(),
        T, C, K, N,
        stream, dtype_code
    );

    // Reduce workspace (T, N, C) → (N, C) via PyTorch's optimized sum kernel
    auto grad_lw = grad_lw_work.sum(0);
    auto grad_lb = grad_lb_work.sum(0);

    // Reshape outputs back
    grad_x = grad_x.view({B, L, N, C});
    grad_sublayer = grad_sublayer.view({B, L, C});

    // Cast float32 grads to param dtype if needed
    if (dtype != torch::kFloat32) {
        grad_lw = grad_lw.to(dtype);
        grad_lb = grad_lb.to(dtype);
    }

    return {grad_x, grad_sublayer, grad_ps, grad_lw, grad_lb, grad_kp};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward",  &mhcv2_fused_forward,  "MHCv2 Fused Forward (CUDA)");
    m.def("fused_backward", &mhcv2_fused_backward, "MHCv2 Fused Backward (CUDA)");
}
