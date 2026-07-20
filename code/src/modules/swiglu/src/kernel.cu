#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__device__ __forceinline__ T sigmoid_func(T x) {
    return 1.0f / (1.0f + std::exp(-static_cast<float>(x)));
}

template <typename scalar_t>
__global__ void swiglu_fwd_kernel(
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ w2,
    scalar_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x1 = static_cast<float>(w1[idx]);
    float x2 = static_cast<float>(w2[idx]);

    float silu_val = x1 / (1.0f + __expf(-x1));
    
    out[idx] = static_cast<scalar_t>(silu_val * x2);
}

template <typename scalar_t>
__global__ void swiglu_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ w2,
    scalar_t* __restrict__ dw1,
    scalar_t* __restrict__ dw2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = static_cast<float>(grad_out[idx]);
    float x1 = static_cast<float>(w1[idx]);
    float x2 = static_cast<float>(w2[idx]);

    float s = 1.0f / (1.0f + __expf(-x1));
    
    float silu_val = x1 * s;
    
    dw2[idx] = static_cast<scalar_t>(g * silu_val);

    float d_silu = s * (1.0f + x1 * (1.0f - s));
    dw1[idx] = static_cast<scalar_t>(g * x2 * d_silu);
}

template <typename scalar_t>
void launch_swiglu_fwd_tmpl(const scalar_t* w1, const scalar_t* w2, scalar_t* out, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_fwd_kernel<scalar_t><<<blocks, threads, 0, s>>>(w1, w2, out, n);
}

template <typename scalar_t>
void launch_swiglu_bwd_tmpl(const scalar_t* g, const scalar_t* w1, const scalar_t* w2, scalar_t* dw1, scalar_t* dw2, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_bwd_kernel<scalar_t><<<blocks, threads, 0, s>>>(g, w1, w2, dw1, dw2, n);
}

void launch_swiglu_fwd_cuda(torch::Tensor w1, torch::Tensor w2, torch::Tensor out) {
    int n = w1.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, w1.scalar_type(), "swiglu_fwd", ([&] {
        launch_swiglu_fwd_tmpl<scalar_t>(
            w1.data_ptr<scalar_t>(),
            w2.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n, stream
        );
    }));
}

void launch_swiglu_bwd_cuda(torch::Tensor g, torch::Tensor w1, torch::Tensor w2, torch::Tensor dw1, torch::Tensor dw2) {
    int n = w1.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, w1.scalar_type(), "swiglu_bwd", ([&] {
        launch_swiglu_bwd_tmpl<scalar_t>(
            g.data_ptr<scalar_t>(),
            w1.data_ptr<scalar_t>(),
            w2.data_ptr<scalar_t>(),
            dw1.data_ptr<scalar_t>(),
            dw2.data_ptr<scalar_t>(),
            n, stream
        );
    }));
}