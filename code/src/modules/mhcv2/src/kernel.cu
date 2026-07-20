/*
 * MHCv2 CUDA Kernel — Fused Residual + LayerNorm + Kronecker Mix
 *
 * Fuses steps 3-5 of MHCv2Layer.forward() into a single kernel per direction:
 *   FWD: res_add → per-stream LayerNorm → Kronecker N×N mix
 *   BWD: reverse chain through all three
 *
 * Template parameter N_STREAMS is the number of streams (2, 4, 8).
 * Each CUDA block processes one (batch, seq_pos) token.
 * Threads collaborate to reduce mean/var over the C dimension.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// ──────────────────────── helpers ────────────────────────

__device__ __forceinline__ float to_float(float v)            { return v; }
__device__ __forceinline__ float to_float(__half v)            { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v)     { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ float          from_float<float>(float v)          { return v; }
template<> __device__ __forceinline__ __half          from_float<__half>(float v)          { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16   from_float<__nv_bfloat16>(float v)   { return __float2bfloat16(v); }

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        val = (lane < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
    }
    __syncthreads();   // prevent race: ensure reads finish before next call's writes
    return val;
}

// ──────────────────────── Kronecker helpers ────────────────────────

// Build N×N doubly-stochastic matrix from K = log2(N) sigmoid probabilities.
// probs[k] in [0,1].  Result stored row-major in M[N*N].
template<int NS>
__device__ __forceinline__ void build_kronecker(const float* probs, float* M, int K) {
    // M = kron(M_{K-1}, ..., M_0)  where M_k = [[p_k, 1-p_k],[1-p_k, p_k]]
    #pragma unroll
    for (int r = 0; r < NS; ++r) {
        #pragma unroll
        for (int c = 0; c < NS; ++c) {
            float val = 1.0f;
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                int bit_r = (r >> k) & 1;
                int bit_c = (c >> k) & 1;
                val *= (bit_r == bit_c) ? probs[k] : (1.0f - probs[k]);
            }
            M[r * NS + c] = val;
        }
    }
}

// ════════════════════════ FORWARD KERNEL ════════════════════════
// One block per (batch, seq_pos) token.
// Input:
//   x_streams:   (B*L, N, C)   — multi-stream tensor
//   sublayer_out:(B*L, C)      — output of attention/FFN sublayer
//   post_scale:  (N,)          — per-stream scaling
//   ln_weight:   (N, C)        — per-stream LN affine weight
//   ln_bias:     (N, C)        — per-stream LN affine bias
//   kron_probs:  (K,)          — sigmoid(kron_logits), pre-computed by caller
// Output:
//   out:         (B*L, N, C)   — result after residual+LN+Kron

template<typename scalar_t, int NS>
__global__ void mhcv2_fwd_kernel(
    const scalar_t* __restrict__ x_streams,    // (T, N, C)
    const scalar_t* __restrict__ sublayer_out,  // (T, C)
    const float*    __restrict__ post_scale,    // (N,)
    const float*    __restrict__ ln_weight,     // (N, C)
    const float*    __restrict__ ln_bias,       // (N, C)
    const float*    __restrict__ kron_probs,    // (K,)
    scalar_t*       __restrict__ out,           // (T, N, C)
    float*          __restrict__ save_mean,     // (T, N) — saved for backward
    float*          __restrict__ save_rstd,     // (T, N) — saved for backward
    int C, int K
) {
    int token = blockIdx.x;          // which (b, l) position
    int tid   = threadIdx.x;

    // Build Kronecker matrix in registers (small: NS*NS ≤ 64 floats)
    float M[NS * NS];
    build_kronecker<NS>(kron_probs, M, K);

    // Load post_scale into registers
    float ps[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) ps[n] = post_scale[n];

    // ── Phase 1: Residual + LayerNorm per stream ──
    // We compute mean and var over C for each stream, then normalize.
    // To amortise the loop, we process all N streams, one element at a time
    // across threads.

    // Shared memory for per-stream mean and rstd
    extern __shared__ float smem[];
    // smem layout: [NS] mean + [NS] rstd
    float* s_mean = smem;
    float* s_rstd = smem + NS;

    // Temporary storage for normalized values — we need to read them twice
    // (once for LN output, once for Kron mix), so we store in shared mem.
    // smem_vals: (NS, C) — too large for shared mem if C is big.
    // Instead, we do two passes: one for stats, one for output.
    // This is acceptable since the data is in L1 cache.

    // Pass 1: compute mean and var for ALL streams in a single pass over C.
    // sublayer_out is read once per element instead of NS times.
    float local_sum[NS];
    float local_sq[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) { local_sum[n] = 0.0f; local_sq[n] = 0.0f; }

    for (int i = tid; i < C; i += blockDim.x) {
        float s_val = to_float(sublayer_out[token * C + i]);  // read once for all streams
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float val = to_float(x_streams[(token * NS + n) * C + i]) + s_val * ps[n];
            local_sum[n] += val;
            local_sq[n]  += val * val;
        }
    }

    // Reduce per stream, reusing shared memory slots sequentially.
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float lsum = blockReduceSum(local_sum[n]);
        __syncthreads();
        float lsq  = blockReduceSum(local_sq[n]);
        __syncthreads();
        if (tid == 0) {
            float mean = lsum / C;
            float var  = lsq / C - mean * mean;
            s_mean[n] = mean;
            s_rstd[n] = rsqrtf(var + 1e-12f);
        }
        __syncthreads();
    }

    // Save mean/rstd for backward
    if (tid < NS) {
        save_mean[token * NS + tid] = s_mean[tid];
        save_rstd[token * NS + tid] = s_rstd[tid];
    }

    // Pass 2: Normalize, apply affine, then Kronecker mix, write output
    // For each output element (n_out, c), we compute:
    //   out[n_out, c] = sum_n M[n_out, n] * (ln_weight[n,c] * (res[n,c] - mean[n]) * rstd[n] + ln_bias[n,c])
    // We process c in parallel across threads.

    for (int c = tid; c < C; c += blockDim.x) {
        // First compute normalized values for all streams at this c
        float s_val = to_float(sublayer_out[token * C + c]);  // hoist: read once for all streams
        float normed[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float x_val = to_float(x_streams[(token * NS + n) * C + c]);
            float res = x_val + s_val * ps[n];
            float norm = (res - s_mean[n]) * s_rstd[n];
            normed[n] = ln_weight[n * C + c] * norm + ln_bias[n * C + c];
        }

        // Kronecker mix: out[n_out, c] = sum_n M[n_out, n] * normed[n]
        #pragma unroll
        for (int n_out = 0; n_out < NS; ++n_out) {
            float acc = 0.0f;
            #pragma unroll
            for (int n = 0; n < NS; ++n) {
                acc += M[n_out * NS + n] * normed[n];
            }
            out[(token * NS + n_out) * C + c] = from_float<scalar_t>(acc);
        }
    }
}


// ════════════════════════ BACKWARD KERNEL ════════════════════════
// Computes gradients for: x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs
//
// One block per token. Each thread processes a subset of C dimensions.
// Scalar gradients (post_scale, kron_probs) use atomicAdd across blocks.

template<typename scalar_t, int NS>
__global__ void mhcv2_bwd_kernel(
    const scalar_t* __restrict__ grad_out,       // (T, N, C)
    const scalar_t* __restrict__ x_streams,      // (T, N, C)
    const scalar_t* __restrict__ sublayer_out,    // (T, C)
    const float*    __restrict__ post_scale,      // (N,)
    const float*    __restrict__ ln_weight,       // (N, C)
    const float*    __restrict__ ln_bias,         // (N, C)
    const float*    __restrict__ kron_probs,      // (K,)
    const float*    __restrict__ save_mean,       // (T, N)
    const float*    __restrict__ save_rstd,       // (T, N)
    scalar_t*       __restrict__ grad_x,          // (T, N, C)
    scalar_t*       __restrict__ grad_sublayer,   // (T, C)
    float*          __restrict__ grad_post_scale, // (N,)  — atomicAdd
    float*          __restrict__ grad_lw_work,    // (T, N, C) — direct write (no atomic)
    float*          __restrict__ grad_lb_work,    // (T, N, C) — direct write (no atomic)
    float*          __restrict__ grad_kron_probs, // (K,)  — atomicAdd
    int C, int K
) {
    int token = blockIdx.x;
    int tid   = threadIdx.x;

    // Rebuild Kronecker matrix
    float M[NS * NS];
    build_kronecker<NS>(kron_probs, M, K);

    float ps[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) ps[n] = post_scale[n];

    // Load saved stats
    float mean[NS], rstd[NS];
    for (int n = 0; n < NS; ++n) {
        mean[n] = save_mean[token * NS + n];
        rstd[n] = save_rstd[token * NS + n];
    }

    // ── Accumulators for scalar gradients (reduced at the end) ──
    float local_grad_ps[NS];
    float local_grad_kp[8]; // max K=8 (N=256, more than enough)
    #pragma unroll
    for (int n = 0; n < NS; ++n) local_grad_ps[n] = 0.0f;
    for (int k = 0; k < K; ++k) local_grad_kp[k] = 0.0f;

    // For LayerNorm backward we need:
    //   sum_c (grad_normed[n,c] * x_hat[n,c])  and  sum_c (grad_normed[n,c])
    // where x_hat = (res - mean) * rstd
    // We accumulate these per stream.
    float sum_gn[NS], sum_gn_xhat[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) { sum_gn[n] = 0.0f; sum_gn_xhat[n] = 0.0f; }

    // ── Pass 1: compute sum_gn and sum_gn_xhat ──
    for (int c = tid; c < C; c += blockDim.x) {
        float s_val = to_float(sublayer_out[token * C + c]);

        float normed_val[NS], xhat[NS], res_val[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float x_val = to_float(x_streams[(token * NS + n) * C + c]);
            res_val[n] = x_val + s_val * ps[n];
            xhat[n] = (res_val[n] - mean[n]) * rstd[n];
            normed_val[n] = ln_weight[n * C + c] * xhat[n] + ln_bias[n * C + c];
        }

        // grad through Kronecker: grad_normed[n] = sum_{n_out} M[n_out, n] * grad_out[n_out]
        float grad_normed[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float acc = 0.0f;
            #pragma unroll
            for (int n_out = 0; n_out < NS; ++n_out) {
                acc += M[n_out * NS + n] * to_float(grad_out[(token * NS + n_out) * C + c]);
            }
            grad_normed[n] = acc;
        }

        // grad through Kron probs
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float gk = 0.0f;
            #pragma unroll
            for (int n_out = 0; n_out < NS; ++n_out) {
                float go = to_float(grad_out[(token * NS + n_out) * C + c]);
                #pragma unroll
                for (int n = 0; n < NS; ++n) {
                    // dM[n_out,n]/dp_k
                    int bit_r = (n_out >> k) & 1;
                    int bit_c = (n >> k) & 1;
                    float sign = (bit_r == bit_c) ? 1.0f : -1.0f;
                    // M[n_out,n] / factor_k * sign
                    float factor_k = (bit_r == bit_c) ? kron_probs[k] : (1.0f - kron_probs[k]);
                    float dM = (factor_k > 1e-12f) ? M[n_out * NS + n] / factor_k * sign : 0.0f;
                    gk += go * normed_val[n] * dM;
                }
            }
            local_grad_kp[k] += gk;
        }

        // grad through LN affine: write per-token to workspace (no atomic contention).
        // Caller reduces workspace over T dimension via PyTorch sum.
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float gn = grad_normed[n];
            grad_lw_work[(token * NS + n) * C + c] = gn * xhat[n];
            grad_lb_work[(token * NS + n) * C + c] = gn;

            // grad through LN: grad_xhat = gn * ln_weight[n,c]
            float g_xhat = gn * ln_weight[n * C + c];
            sum_gn[n] += g_xhat;
            sum_gn_xhat[n] += g_xhat * xhat[n];
        }
    }

    // Reduce sum_gn and sum_gn_xhat across threads
    for (int n = 0; n < NS; ++n) {
        sum_gn[n] = blockReduceSum(sum_gn[n]);
        __syncthreads();
        sum_gn_xhat[n] = blockReduceSum(sum_gn_xhat[n]);
        __syncthreads();
    }

    // Broadcast reduced sums via shared memory
    extern __shared__ float smem[];
    float* s_sum_gn     = smem;
    float* s_sum_gn_xh  = smem + NS;
    if (tid == 0) {
        for (int n = 0; n < NS; ++n) {
            s_sum_gn[n]    = sum_gn[n];
            s_sum_gn_xh[n] = sum_gn_xhat[n];
        }
    }
    __syncthreads();

    // ── Pass 2: compute per-element gradients ──
    for (int c = tid; c < C; c += blockDim.x) {
        float s_val = to_float(sublayer_out[token * C + c]);
        float grad_sublayer_c = 0.0f;

        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float x_val = to_float(x_streams[(token * NS + n) * C + c]);
            float res = x_val + s_val * ps[n];
            float xhat = (res - mean[n]) * rstd[n];

            // Recompute grad_normed for this c
            float grad_normed_n = 0.0f;
            #pragma unroll
            for (int n_out = 0; n_out < NS; ++n_out) {
                grad_normed_n += M[n_out * NS + n] * to_float(grad_out[(token * NS + n_out) * C + c]);
            }

            float g_xhat = grad_normed_n * ln_weight[n * C + c];

            // LayerNorm backward: grad_res = rstd * (g_xhat - (sum_gn + xhat * sum_gn_xhat) / C)
            float grad_res = rstd[n] * (g_xhat - (s_sum_gn[n] + xhat * s_sum_gn_xh[n]) / C);

            // grad_x = grad_res
            grad_x[(token * NS + n) * C + c] = from_float<scalar_t>(grad_res);

            // grad_sublayer += grad_res * post_scale[n]
            grad_sublayer_c += grad_res * ps[n];

            // grad_post_scale[n] += grad_res * sublayer_out[c]
            local_grad_ps[n] += grad_res * s_val;
        }

        // Accumulate sublayer gradient
        // Multiple streams contribute — write per-c (each c only written once)
        grad_sublayer[token * C + c] = from_float<scalar_t>(grad_sublayer_c);
    }

    // ── Reduce and atomicAdd scalar gradients ──
    for (int n = 0; n < NS; ++n) {
        local_grad_ps[n] = blockReduceSum(local_grad_ps[n]);
        __syncthreads();
    }
    for (int k = 0; k < K; ++k) {
        local_grad_kp[k] = blockReduceSum(local_grad_kp[k]);
        __syncthreads();
    }

    if (tid == 0) {
        for (int n = 0; n < NS; ++n)
            atomicAdd(&grad_post_scale[n], local_grad_ps[n]);
        for (int k = 0; k < K; ++k)
            atomicAdd(&grad_kron_probs[k], local_grad_kp[k]);
    }
}


// ════════════════════════ LAUNCHER FUNCTIONS ════════════════════════

// Forward dispatch by dtype and N
template<int NS>
void launch_mhcv2_fwd_typed(
    const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs,
    void* out, float* save_mean, float* save_rstd,
    int total_tokens, int C, int K, cudaStream_t stream, int dtype_code
) {
    int threads = min(256, ((C + 31) / 32) * 32);
    int smem = 2 * NS * sizeof(float);

    #define LAUNCH_FWD(T) \
        mhcv2_fwd_kernel<T, NS><<<total_tokens, threads, smem, stream>>>( \
            (const T*)x_streams, (const T*)sublayer_out, \
            post_scale, ln_weight, ln_bias, kron_probs, \
            (T*)out, save_mean, save_rstd, C, K)

    if (dtype_code == 1)      LAUNCH_FWD(__half);
    else if (dtype_code == 2) LAUNCH_FWD(__nv_bfloat16);
    else                      LAUNCH_FWD(float);

    #undef LAUNCH_FWD
}

// Backward dispatch
template<int NS>
void launch_mhcv2_bwd_typed(
    const void* grad_out, const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs, const float* save_mean, const float* save_rstd,
    void* grad_x, void* grad_sublayer,
    float* grad_post_scale, float* grad_lw_work, float* grad_lb_work,
    float* grad_kron_probs,
    int total_tokens, int C, int K, cudaStream_t stream, int dtype_code
) {
    int threads = min(256, ((C + 31) / 32) * 32);
    int smem = 2 * NS * sizeof(float);

    #define LAUNCH_BWD(T) \
        mhcv2_bwd_kernel<T, NS><<<total_tokens, threads, smem, stream>>>( \
            (const T*)grad_out, (const T*)x_streams, (const T*)sublayer_out, \
            post_scale, ln_weight, ln_bias, kron_probs, save_mean, save_rstd, \
            (T*)grad_x, (T*)grad_sublayer, \
            grad_post_scale, grad_lw_work, grad_lb_work, grad_kron_probs, \
            C, K)

    if (dtype_code == 1)      LAUNCH_BWD(__half);
    else if (dtype_code == 2) LAUNCH_BWD(__nv_bfloat16);
    else                      LAUNCH_BWD(float);

    #undef LAUNCH_BWD
}

// Public C-linkage launchers dispatching on N_STREAMS
void launch_mhcv2_fwd(
    const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs,
    void* out, float* save_mean, float* save_rstd,
    int total_tokens, int C, int K, int N_STREAMS,
    cudaStream_t stream, int dtype_code
) {
    switch (N_STREAMS) {
        case 2: launch_mhcv2_fwd_typed<2>(x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, out, save_mean, save_rstd, total_tokens, C, K, stream, dtype_code); break;
        case 4: launch_mhcv2_fwd_typed<4>(x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, out, save_mean, save_rstd, total_tokens, C, K, stream, dtype_code); break;
        case 8: launch_mhcv2_fwd_typed<8>(x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, out, save_mean, save_rstd, total_tokens, C, K, stream, dtype_code); break;
        default: break; // caller should validate
    }
}

void launch_mhcv2_bwd(
    const void* grad_out, const void* x_streams, const void* sublayer_out,
    const float* post_scale, const float* ln_weight, const float* ln_bias,
    const float* kron_probs, const float* save_mean, const float* save_rstd,
    void* grad_x, void* grad_sublayer,
    float* grad_post_scale, float* grad_lw_work, float* grad_lb_work,
    float* grad_kron_probs,
    int total_tokens, int C, int K, int N_STREAMS,
    cudaStream_t stream, int dtype_code
) {
    switch (N_STREAMS) {
        case 2: launch_mhcv2_bwd_typed<2>(grad_out, x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, save_mean, save_rstd, grad_x, grad_sublayer, grad_post_scale, grad_lw_work, grad_lb_work, grad_kron_probs, total_tokens, C, K, stream, dtype_code); break;
        case 4: launch_mhcv2_bwd_typed<4>(grad_out, x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, save_mean, save_rstd, grad_x, grad_sublayer, grad_post_scale, grad_lw_work, grad_lb_work, grad_kron_probs, total_tokens, C, K, stream, dtype_code); break;
        case 8: launch_mhcv2_bwd_typed<8>(grad_out, x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_probs, save_mean, save_rstd, grad_x, grad_sublayer, grad_post_scale, grad_lw_work, grad_lb_work, grad_kron_probs, total_tokens, C, K, stream, dtype_code); break;
        default: break;
    }
}
