#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define EPSILON 1e-8f
#define N 4

__device__ __forceinline__ void sinkhorn_inplace_N4(float matrix[N][N], int n_iters) {
    #pragma unroll
    for (int iter = 0; iter < n_iters; ++iter) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float max_val = fmaxf(fmaxf(matrix[i][0], matrix[i][1]), fmaxf(matrix[i][2], matrix[i][3]));
            float sum_exp = __expf(matrix[i][0] - max_val) + __expf(matrix[i][1] - max_val) + 
                           __expf(matrix[i][2] - max_val) + __expf(matrix[i][3] - max_val);
            float lse = max_val + __logf(sum_exp + EPSILON);
            #pragma unroll
            for (int j = 0; j < N; ++j) matrix[i][j] -= lse;
        }
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            float max_val = fmaxf(fmaxf(matrix[0][j], matrix[1][j]), fmaxf(matrix[2][j], matrix[3][j]));
            float sum_exp = __expf(matrix[0][j] - max_val) + __expf(matrix[1][j] - max_val) + 
                           __expf(matrix[2][j] - max_val) + __expf(matrix[3][j] - max_val);
            float lse = max_val + __logf(sum_exp + EPSILON);
            #pragma unroll
            for (int i = 0; i < N; ++i) matrix[i][j] -= lse;
        }
    }
}

__global__ void rms_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ x_norm,
    float* __restrict__ rstd_out,
    float eps,
    int total_tokens,
    int dim
) {
    extern __shared__ float shared[];
    
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int tid = threadIdx.x;
    int offset = token_idx * dim;
    
    float local_sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[offset + i];
        local_sum_sq += v * v;
    }
    
    shared[tid] = local_sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    float rstd = rsqrtf(shared[0] / dim + eps);
    if (tid == 0) rstd_out[token_idx] = rstd;
    __syncthreads();
    
    for (int i = tid; i < dim; i += blockDim.x) {
        x_norm[offset + i] = x[offset + i] * rstd * weight[i];
    }
}

__global__ void rms_norm_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ rstd,
    const float* __restrict__ weight,
    float* __restrict__ grad_x,
    int total_tokens,
    int dim
) {
    extern __shared__ float shared[];
    
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int tid = threadIdx.x;
    int offset = token_idx * dim;
    float rstd_val = rstd[token_idx];
    
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        local_sum += grad_out[offset + i] * weight[i] * x[offset + i];
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    float scale = shared[0] * rstd_val * rstd_val / dim;
    __syncthreads();
    
    for (int i = tid; i < dim; i += blockDim.x) {
        float g = grad_out[offset + i] * weight[i];
        grad_x[offset + i] = (g - scale * x[offset + i]) * rstd_val;
    }
}

__global__ void fused_act_sinkhorn_fwd_kernel(
    const float* __restrict__ proj_output,
    const float* __restrict__ bias,
    float alpha_pre, float alpha_post, float alpha_res,
    int n_iters,
    float* __restrict__ H_pre,
    float* __restrict__ H_post,
    float* __restrict__ H_res,
    int total_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_items) return;

    int out_dim = N + N + N * N;
    int base_proj = idx * out_dim;
    int base_n = idx * N;
    int base_nn = idx * N * N;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float val = alpha_pre * proj_output[base_proj + i] + bias[i];
        H_pre[base_n + i] = 1.0f / (1.0f + __expf(-val));
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float val = alpha_post * proj_output[base_proj + N + i] + bias[N + i];
        H_post[base_n + i] = 2.0f / (1.0f + __expf(-val));
    }

    float matrix[N][N];
    int res_start = 2 * N;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = alpha_res * proj_output[base_proj + res_start + i * N + j] + bias[res_start + i * N + j];
        }
    }

    sinkhorn_inplace_N4(matrix, n_iters);

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            H_res[base_nn + i * N + j] = __expf(matrix[i][j]);
        }
    }
}

__global__ void fused_act_sinkhorn_bwd_kernel(
    const float* __restrict__ grad_H_pre,
    const float* __restrict__ grad_H_post,
    const float* __restrict__ grad_H_res,
    const float* __restrict__ proj_output,
    const float* __restrict__ bias,
    const float* __restrict__ H_pre,
    const float* __restrict__ H_post,
    float alpha_pre, float alpha_post, float alpha_res,
    int n_iters,
    float* __restrict__ grad_proj_output,
    float* __restrict__ grad_bias,
    float* __restrict__ grad_alpha_pre,
    float* __restrict__ grad_alpha_post,
    float* __restrict__ grad_alpha_res,
    int total_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_items) return;

    int out_dim = N + N + N * N;
    int base_proj = idx * out_dim;
    int base_n = idx * N;
    int base_nn = idx * N * N;

    float local_g_alpha_pre = 0.0f;
    float local_g_alpha_post = 0.0f;
    float local_g_alpha_res = 0.0f;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float h = H_pre[base_n + i];
        float g = grad_H_pre[base_n + i];
        float dsig = h * (1.0f - h);
        float grad_val = g * dsig;
        grad_proj_output[base_proj + i] = grad_val * alpha_pre;
        atomicAdd(&grad_bias[i], grad_val);
        local_g_alpha_pre += grad_val * proj_output[base_proj + i];
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float h = H_post[base_n + i] * 0.5f;
        float g = grad_H_post[base_n + i];
        float dsig = h * (1.0f - h);
        float grad_val = g * 2.0f * dsig;
        grad_proj_output[base_proj + N + i] = grad_val * alpha_post;
        atomicAdd(&grad_bias[N + i], grad_val);
        local_g_alpha_post += grad_val * proj_output[base_proj + N + i];
    }

    float matrix[N][N];
    int res_start = 2 * N;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = alpha_res * proj_output[base_proj + res_start + i * N + j] + bias[res_start + i * N + j];
        }
    }

    sinkhorn_inplace_N4(matrix, n_iters);

    float grad_matrix[N][N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            grad_matrix[i][j] = grad_H_res[base_nn + i * N + j] * __expf(matrix[i][j]);
        }
    }

    for (int iter = n_iters - 1; iter >= 0; --iter) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float sum_g = grad_matrix[i][0] + grad_matrix[i][1] + grad_matrix[i][2] + grad_matrix[i][3];
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                grad_matrix[i][j] -= __expf(matrix[i][j]) * sum_g;
            }
        }
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            float sum_g = grad_matrix[0][j] + grad_matrix[1][j] + grad_matrix[2][j] + grad_matrix[3][j];
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                grad_matrix[i][j] -= __expf(matrix[i][j]) * sum_g;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            float val = grad_matrix[i][j];
            grad_proj_output[base_proj + res_start + i * N + j] = val * alpha_res;
            atomicAdd(&grad_bias[res_start + i * N + j], val);
            local_g_alpha_res += val * proj_output[base_proj + res_start + i * N + j];
        }
    }

    atomicAdd(grad_alpha_pre, local_g_alpha_pre);
    atomicAdd(grad_alpha_post, local_g_alpha_post);
    atomicAdd(grad_alpha_res, local_g_alpha_res);
}

void launch_rms_fwd(const float* x, const float* w, float* xn, float* rstd, float eps, int total, int dim, cudaStream_t stream) {
    int threads = min(256, dim);
    int smem = threads * sizeof(float);
    rms_norm_fwd_kernel<<<total, threads, smem, stream>>>(x, w, xn, rstd, eps, total, dim);
}

void launch_rms_bwd(const float* dout, const float* x, const float* rstd, const float* w, float* dx, int total, int dim, cudaStream_t stream) {
    int threads = min(256, dim);
    int smem = threads * sizeof(float);
    rms_norm_bwd_kernel<<<total, threads, smem, stream>>>(dout, x, rstd, w, dx, total, dim);
}

void launch_fused_act_fwd(const float* proj, const float* bias, float a_pre, float a_post, float a_res, int iters, float* h_pre, float* h_post, float* h_res, int total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_act_sinkhorn_fwd_kernel<<<blocks, threads, 0, stream>>>(proj, bias, a_pre, a_post, a_res, iters, h_pre, h_post, h_res, total);
}

void launch_fused_act_bwd(const float* gh_pre, const float* gh_post, const float* gh_res, const float* proj, const float* bias, const float* h_pre, const float* h_post, float a_pre, float a_post, float a_res, int iters, float* g_proj, float* g_bias, float* ga_pre, float* ga_post, float* ga_res, int total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_act_sinkhorn_bwd_kernel<<<blocks, threads, 0, stream>>>(gh_pre, gh_post, gh_res, proj, bias, h_pre, h_post, a_pre, a_post, a_res, iters, g_proj, g_bias, ga_pre, ga_post, ga_res, total);
}