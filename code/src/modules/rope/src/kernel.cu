#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <typename scalar_t>
__global__ void rope_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ cos_buf,
    const scalar_t* __restrict__ sin_buf,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    int B, int L, int H, int D,
    int head_dim_stride,
    int seq_stride,
    int batch_stride
) {
    int half_d = D / 2;
    int tid = threadIdx.x;
    
    if (tid >= half_d) return;

    int l_idx = blockIdx.x;
    int bh_idx = blockIdx.y;

    int h_idx = bh_idx % H;
    int b_idx = bh_idx / H;

    int data_offset = b_idx * batch_stride + l_idx * seq_stride + h_idx * head_dim_stride;
    int cos_sin_offset = l_idx * D + tid;

    float c = to_float(cos_buf[cos_sin_offset]);
    float s = to_float(sin_buf[cos_sin_offset]);

    int idx1 = data_offset + tid;
    int idx2 = data_offset + tid + half_d;

    float q1 = to_float(q[idx1]);
    float q2 = to_float(q[idx2]);

    float k1 = to_float(k[idx1]);
    float k2 = to_float(k[idx2]);

    q_out[idx1] = static_cast<scalar_t>(q1 * c - q2 * s);
    q_out[idx2] = static_cast<scalar_t>(q2 * c + q1 * s);

    k_out[idx1] = static_cast<scalar_t>(k1 * c - k2 * s);
    k_out[idx2] = static_cast<scalar_t>(k2 * c + k1 * s);
}

template <typename scalar_t>
__global__ void rope_bwd_kernel(
    const scalar_t* __restrict__ g_q_out,
    const scalar_t* __restrict__ g_k_out,
    const scalar_t* __restrict__ cos_buf,
    const scalar_t* __restrict__ sin_buf,
    scalar_t* __restrict__ g_q,
    scalar_t* __restrict__ g_k,
    int B, int L, int H, int D,
    int head_dim_stride, int seq_stride, int batch_stride
) {
    int half_d = D / 2;
    int tid = threadIdx.x;
    if (tid >= half_d) return;

    int l_idx = blockIdx.x;
    int h_idx = blockIdx.y % H;
    int b_idx = blockIdx.y / H;

    int data_offset = b_idx * batch_stride + l_idx * seq_stride + h_idx * head_dim_stride;
    int cos_sin_offset = l_idx * D + tid;

    float c = to_float(cos_buf[cos_sin_offset]);
    float s = to_float(sin_buf[cos_sin_offset]);

    int idx1 = data_offset + tid;
    int idx2 = data_offset + tid + half_d;

    float gq1 = to_float(g_q_out[idx1]);
    float gq2 = to_float(g_q_out[idx2]);
    float gk1 = to_float(g_k_out[idx1]);
    float gk2 = to_float(g_k_out[idx2]);

    g_q[idx1] = static_cast<scalar_t>(gq1 * c + gq2 * s);
    g_q[idx2] = static_cast<scalar_t>(gq2 * c - gq1 * s);

    g_k[idx1] = static_cast<scalar_t>(gk1 * c + gk2 * s);
    g_k[idx2] = static_cast<scalar_t>(gk2 * c - gk1 * s);
}

template <typename scalar_t>
void launch_rope_fwd_tmpl(const scalar_t* q, const scalar_t* k, const scalar_t* c, const scalar_t* s, scalar_t* qo, scalar_t* ko, int B, int L, int H, int D, cudaStream_t stream) {
    dim3 grid(L, B * H); 
    int half_d = D / 2;
    int threads = (half_d > 1024) ? 1024 : half_d; 
    int head_stride = D;
    int seq_stride = H * D;
    int batch_stride = L * H * D;

    rope_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(q, k, c, s, qo, ko, B, L, H, D, head_stride, seq_stride, batch_stride);
}

template <typename scalar_t>
void launch_rope_bwd_tmpl(const scalar_t* gq, const scalar_t* gk, const scalar_t* c, const scalar_t* s, scalar_t* dq, scalar_t* dk, int B, int L, int H, int D, cudaStream_t stream) {
    dim3 grid(L, B * H);
    int half_d = D / 2;
    int threads = (half_d > 1024) ? 1024 : half_d;
    
    int head_stride = D;
    int seq_stride = H * D;
    int batch_stride = L * H * D;

    rope_bwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(gq, gk, c, s, dq, dk, B, L, H, D, head_stride, seq_stride, batch_stride);
}

void launch_rope_fwd_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out) {
    int B = q.size(0);
    int L = q.size(1);
    int H = q.size(2);
    int D = q.size(3);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "rope_fwd", ([&] {
        launch_rope_fwd_tmpl<scalar_t>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),
            q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(),
            B, L, H, D, at::cuda::getCurrentCUDAStream()
        );
    }));
}

void launch_rope_bwd_cuda(torch::Tensor gq_in, torch::Tensor gk_in, torch::Tensor cos, torch::Tensor sin, torch::Tensor gq, torch::Tensor gk) {
    int B = gq_in.size(0);
    int L = gq_in.size(1);
    int H = gq_in.size(2);
    int D = gq_in.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gq_in.scalar_type(), "rope_bwd", ([&] {
        launch_rope_bwd_tmpl<scalar_t>(
            gq_in.data_ptr<scalar_t>(), gk_in.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),
            gq.data_ptr<scalar_t>(), gk.data_ptr<scalar_t>(),
            B, L, H, D, at::cuda::getCurrentCUDAStream()
        );
    }));
}