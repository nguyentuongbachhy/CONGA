#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#define EPSILON 1e-8f
constexpr int N = 4;
constexpr int OUT_DIM = N + N + 2;  // 4 pre + 4 post + 2 Kronecker factors

__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ __half from_float<__half>(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) { return __float2bfloat16(val); }

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
    }
    return val;
}

template<int SIZE>
__device__ __forceinline__ void blockReduceVec(float* val) {
    __shared__ float shared[SIZE][32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    #pragma unroll
    for (int i = 0; i < SIZE; ++i) {
        val[i] = warpReduceSum(val[i]);
        if (lane == 0) shared[i][wid] = val[i];
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        #pragma unroll
        for (int i = 0; i < SIZE; ++i) {
            val[i] = (lane < num_warps) ? shared[i][lane] : 0.0f;
            if (wid == 0) val[i] = warpReduceSum(val[i]);
        }
    }
}

// KromHC: construct 4×4 doubly-stochastic matrix via Kronecker product of two 2×2 factors.
// M1(a) = [[a, 1-a], [1-a, a]]
// M2(b) = [[b, 1-b], [1-b, b]]
// H_res = M1 ⊗ M2  (exact double stochasticity by construction)
__device__ __forceinline__ void kronecker_construct_N4(float a, float b, float matrix[N][N]) {
    #pragma unroll
    for (int r = 0; r < N; ++r) {
        #pragma unroll
        for (int c = 0; c < N; ++c) {
            float m1 = ((r >> 1) == (c >> 1)) ? a : (1.0f - a);
            float m2 = ((r & 1) == (c & 1)) ? b : (1.0f - b);
            matrix[r][c] = m1 * m2;
        }
    }
}

template<typename scalar_t>
__global__ void compute_rstd_kernel(
    const scalar_t* __restrict__ x,
    float* __restrict__ rstd_out,
    float eps,
    int total_tokens,
    int dim
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int offset = token_idx * dim;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = to_float(x[offset + i]);
        local_sum_sq += v * v;
    }

    local_sum_sq = blockReduceSum(local_sum_sq);

    if (threadIdx.x == 0) {
        rstd_out[token_idx] = rsqrtf(local_sum_sq / dim + eps);
    }
}

template<typename scalar_t>
__global__ void fused_act_krom_fwd_kernel(
    const scalar_t* __restrict__ proj_raw,
    const scalar_t* __restrict__ bias,
    const float* __restrict__ rstd,
    const float* __restrict__ alpha_pre_ptr,
    const float* __restrict__ alpha_post_ptr,
    const float* __restrict__ alpha_res_ptr,
    scalar_t* __restrict__ H_pre,
    scalar_t* __restrict__ H_post,
    scalar_t* __restrict__ H_res,
    int total_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_items) return;

    float alpha_pre = *alpha_pre_ptr;
    float alpha_post = *alpha_post_ptr;
    float alpha_res = *alpha_res_ptr;

    int base_proj = idx * OUT_DIM;
    int base_n = idx * N;
    int base_nn = idx * N * N;
    float rstd_val = rstd[idx];

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float proj_val = to_float(proj_raw[base_proj + i]) * rstd_val;
        float bias_val = bias ? to_float(bias[i]) : 0.0f;
        float val = alpha_pre * proj_val + bias_val;
        H_pre[base_n + i] = from_float<scalar_t>(1.0f / (1.0f + __expf(-val)));
    }

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float proj_val = to_float(proj_raw[base_proj + N + i]) * rstd_val;
        float bias_val = bias ? to_float(bias[N + i]) : 0.0f;
        float val = alpha_post * proj_val + bias_val;
        H_post[base_n + i] = from_float<scalar_t>(2.0f / (1.0f + __expf(-val)));
    }

    // Kronecker factors: 2 scalars -> exact 4×4 doubly-stochastic H_res
    int res_start = 2 * N;
    float raw_a = to_float(proj_raw[base_proj + res_start]) * rstd_val;
    float raw_b = to_float(proj_raw[base_proj + res_start + 1]) * rstd_val;
    float bias_a = bias ? to_float(bias[res_start]) : 0.0f;
    float bias_b = bias ? to_float(bias[res_start + 1]) : 0.0f;
    float a = 1.0f / (1.0f + __expf(-(alpha_res * raw_a + bias_a)));
    float b = 1.0f / (1.0f + __expf(-(alpha_res * raw_b + bias_b)));

    float matrix[N][N];
    kronecker_construct_N4(a, b, matrix);

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            H_res[base_nn + i * N + j] = from_float<scalar_t>(matrix[i][j]);
        }
    }
}

template<typename scalar_t>
__global__ void fused_act_krom_bwd_kernel(
    const scalar_t* __restrict__ grad_H_pre,
    const scalar_t* __restrict__ grad_H_post,
    const scalar_t* __restrict__ grad_H_res,
    const scalar_t* __restrict__ proj_raw,
    const scalar_t* __restrict__ bias,
    const float* __restrict__ rstd,
    const scalar_t* __restrict__ H_pre,
    const scalar_t* __restrict__ H_post,
    const float* __restrict__ alpha_pre_ptr,
    const float* __restrict__ alpha_post_ptr,
    const float* __restrict__ alpha_res_ptr,
    scalar_t* __restrict__ grad_proj_raw,
    float* __restrict__ grad_bias,
    float* __restrict__ grad_alpha_pre,
    float* __restrict__ grad_alpha_post,
    float* __restrict__ grad_alpha_res,
    float* __restrict__ grad_rstd_out,
    int total_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float alpha_pre = *alpha_pre_ptr;
    float alpha_post = *alpha_post_ptr;
    float alpha_res = *alpha_res_ptr;

    float local_g_alpha_pre = 0.0f;
    float local_g_alpha_post = 0.0f;
    float local_g_alpha_res = 0.0f;
    float local_g_rstd = 0.0f;
    float local_g_bias[OUT_DIM];

    #pragma unroll
    for (int i = 0; i < OUT_DIM; ++i) local_g_bias[i] = 0.0f;

    if (idx < total_items) {
        int base_proj = idx * OUT_DIM;
        int base_n = idx * N;
        int base_nn = idx * N * N;
        float rstd_val = rstd[idx];

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float h = to_float(H_pre[base_n + i]);
            float g = to_float(grad_H_pre[base_n + i]);
            float dsig = h * (1.0f - h);
            float grad_val = g * dsig;
            float raw_val = to_float(proj_raw[base_proj + i]);
            float proj_val = raw_val * rstd_val;

            grad_proj_raw[base_proj + i] = from_float<scalar_t>(grad_val * alpha_pre * rstd_val);
            local_g_bias[i] = grad_val;
            local_g_alpha_pre += grad_val * proj_val;
            local_g_rstd += grad_val * alpha_pre * raw_val;
        }

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float h = to_float(H_post[base_n + i]) * 0.5f;
            float g = to_float(grad_H_post[base_n + i]);
            float dsig = h * (1.0f - h);
            float grad_val = g * 2.0f * dsig;
            float raw_val = to_float(proj_raw[base_proj + N + i]);
            float proj_val = raw_val * rstd_val;

            grad_proj_raw[base_proj + N + i] = from_float<scalar_t>(grad_val * alpha_post * rstd_val);
            local_g_bias[N + i] = grad_val;
            local_g_alpha_post += grad_val * proj_val;
            local_g_rstd += grad_val * alpha_post * raw_val;
        }

        // Recompute a, b from saved proj_raw
        int res_start = 2 * N;
        float raw_a_proj = to_float(proj_raw[base_proj + res_start]);
        float raw_b_proj = to_float(proj_raw[base_proj + res_start + 1]);
        float bias_a = bias ? to_float(bias[res_start]) : 0.0f;
        float bias_b = bias ? to_float(bias[res_start + 1]) : 0.0f;
        float a = 1.0f / (1.0f + __expf(-(alpha_res * raw_a_proj * rstd_val + bias_a)));
        float b = 1.0f / (1.0f + __expf(-(alpha_res * raw_b_proj * rstd_val + bias_b)));
        float dsig_a = a * (1.0f - a);
        float dsig_b = b * (1.0f - b);

        // Accumulate grad_a and grad_b from grad_H_res via Kronecker backward
        float grad_a = 0.0f, grad_b = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            #pragma unroll
            for (int c = 0; c < N; ++c) {
                float g = to_float(grad_H_res[base_nn + r * N + c]);
                float sign_m1 = ((r >> 1) == (c >> 1)) ? 1.0f : -1.0f;
                float sign_m2 = ((r & 1) == (c & 1)) ? 1.0f : -1.0f;
                float m2_val  = ((r & 1) == (c & 1)) ? b : (1.0f - b);
                float m1_val  = ((r >> 1) == (c >> 1)) ? a : (1.0f - a);
                grad_a += g * sign_m1 * m2_val;
                grad_b += g * m1_val  * sign_m2;
            }
        }

        // Chain rule through sigmoid and alpha_res
        float ga_da = grad_a * dsig_a;
        float gb_db = grad_b * dsig_b;

        grad_proj_raw[base_proj + res_start]     = from_float<scalar_t>(ga_da * alpha_res * rstd_val);
        grad_proj_raw[base_proj + res_start + 1] = from_float<scalar_t>(gb_db * alpha_res * rstd_val);
        local_g_bias[res_start]     = ga_da;
        local_g_bias[res_start + 1] = gb_db;
        local_g_alpha_res += ga_da * raw_a_proj * rstd_val + gb_db * raw_b_proj * rstd_val;
        local_g_rstd      += alpha_res * (ga_da * raw_a_proj + gb_db * raw_b_proj);

        grad_rstd_out[idx] = local_g_rstd;
    }

    local_g_alpha_pre  = blockReduceSum(local_g_alpha_pre);
    local_g_alpha_post = blockReduceSum(local_g_alpha_post);
    local_g_alpha_res  = blockReduceSum(local_g_alpha_res);
    blockReduceVec<OUT_DIM>(local_g_bias);

    if (threadIdx.x == 0) {
        atomicAdd(grad_alpha_pre,  local_g_alpha_pre);
        atomicAdd(grad_alpha_post, local_g_alpha_post);
        atomicAdd(grad_alpha_res,  local_g_alpha_res);
        if (grad_bias) {
            #pragma unroll
            for (int i = 0; i < OUT_DIM; ++i) {
                atomicAdd(&grad_bias[i], local_g_bias[i]);
            }
        }
    }
}

template<typename scalar_t>
__global__ void apply_rstd_grad_kernel(
    const scalar_t* __restrict__ grad_x_matmul,
    const float* __restrict__ grad_rstd,
    const float* __restrict__ rstd,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_x,
    int total_tokens,
    int dim
) {
    for (int token_idx = blockIdx.x; token_idx < total_tokens; token_idx += gridDim.x) {
        float rstd_val = rstd[token_idx];
        float grad_rstd_val = grad_rstd[token_idx];
        float scale = -grad_rstd_val * rstd_val * rstd_val * rstd_val / (float)dim;

        int offset = token_idx * dim;

        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float g_matmul = to_float(grad_x_matmul[offset + i]);
            float x_val = to_float(x[offset + i]);
            grad_x[offset + i] = from_float<scalar_t>(g_matmul + scale * x_val);
        }
    }
}

void launch_compute_rstd(const void* x, float* rstd, float eps, int total, int dim, cudaStream_t stream, int dtype_code) {
    int threads = min(256, (dim + 31) & ~31);

    if (dtype_code == 1) {
        compute_rstd_kernel<__half><<<total, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(x), rstd, eps, total, dim);
    } else if (dtype_code == 2) {
        compute_rstd_kernel<__nv_bfloat16><<<total, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), rstd, eps, total, dim);
    } else {
        compute_rstd_kernel<float><<<total, threads, 0, stream>>>(
            reinterpret_cast<const float*>(x), rstd, eps, total, dim);
    }
}

void launch_fused_act_fwd(const void* proj_raw, const void* bias, const float* rstd, const float* a_pre, const float* a_post, const float* a_res, void* h_pre, void* h_post, void* h_res, int total, cudaStream_t stream, int dtype_code) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (dtype_code == 1) {
        fused_act_krom_fwd_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(proj_raw), reinterpret_cast<const __half*>(bias),
            rstd, a_pre, a_post, a_res,
            reinterpret_cast<__half*>(h_pre), reinterpret_cast<__half*>(h_post), reinterpret_cast<__half*>(h_res), total);
    } else if (dtype_code == 2) {
        fused_act_krom_fwd_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(proj_raw), reinterpret_cast<const __nv_bfloat16*>(bias),
            rstd, a_pre, a_post, a_res,
            reinterpret_cast<__nv_bfloat16*>(h_pre), reinterpret_cast<__nv_bfloat16*>(h_post), reinterpret_cast<__nv_bfloat16*>(h_res), total);
    } else {
        fused_act_krom_fwd_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(proj_raw), reinterpret_cast<const float*>(bias),
            rstd, a_pre, a_post, a_res,
            reinterpret_cast<float*>(h_pre), reinterpret_cast<float*>(h_post), reinterpret_cast<float*>(h_res), total);
    }
}

void launch_fused_act_bwd(const void* gh_pre, const void* gh_post, const void* gh_res, const void* proj_raw, const void* bias, const float* rstd, const void* h_pre, const void* h_post, const float* a_pre, const float* a_post, const float* a_res, void* g_proj_raw, float* g_bias, float* ga_pre, float* ga_post, float* ga_res, float* g_rstd, int total, cudaStream_t stream, int dtype_code) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (dtype_code == 1) {
        fused_act_krom_bwd_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(gh_pre), reinterpret_cast<const __half*>(gh_post), reinterpret_cast<const __half*>(gh_res),
            reinterpret_cast<const __half*>(proj_raw), reinterpret_cast<const __half*>(bias), rstd,
            reinterpret_cast<const __half*>(h_pre), reinterpret_cast<const __half*>(h_post),
            a_pre, a_post, a_res, reinterpret_cast<__half*>(g_proj_raw),
            g_bias, ga_pre, ga_post, ga_res, g_rstd, total);
    } else if (dtype_code == 2) {
        fused_act_krom_bwd_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(gh_pre), reinterpret_cast<const __nv_bfloat16*>(gh_post), reinterpret_cast<const __nv_bfloat16*>(gh_res),
            reinterpret_cast<const __nv_bfloat16*>(proj_raw), reinterpret_cast<const __nv_bfloat16*>(bias), rstd,
            reinterpret_cast<const __nv_bfloat16*>(h_pre), reinterpret_cast<const __nv_bfloat16*>(h_post),
            a_pre, a_post, a_res, reinterpret_cast<__nv_bfloat16*>(g_proj_raw),
            g_bias, ga_pre, ga_post, ga_res, g_rstd, total);
    } else {
        fused_act_krom_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(gh_pre), reinterpret_cast<const float*>(gh_post), reinterpret_cast<const float*>(gh_res),
            reinterpret_cast<const float*>(proj_raw), reinterpret_cast<const float*>(bias), rstd,
            reinterpret_cast<const float*>(h_pre), reinterpret_cast<const float*>(h_post),
            a_pre, a_post, a_res, reinterpret_cast<float*>(g_proj_raw),
            g_bias, ga_pre, ga_post, ga_res, g_rstd, total);
    }
}

void launch_apply_rstd_grad(const void* grad_x_matmul, const float* grad_rstd, const float* rstd, const void* x, void* grad_x, int total, int dim, cudaStream_t stream, int dtype_code) {
    int threads = min(256, (dim + 31) & ~31);
    int blocks = min(8192, total);

    if (dtype_code == 1) {
        apply_rstd_grad_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_x_matmul), grad_rstd, rstd,
            reinterpret_cast<const __half*>(x), reinterpret_cast<__half*>(grad_x), total, dim);
    } else if (dtype_code == 2) {
        apply_rstd_grad_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_x_matmul), grad_rstd, rstd,
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(grad_x), total, dim);
    } else {
        apply_rstd_grad_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(grad_x_matmul), grad_rstd, rstd,
            reinterpret_cast<const float*>(x), reinterpret_cast<float*>(grad_x), total, dim);
    }
}
