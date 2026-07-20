#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- Helper Functions ---
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ __half from_float<__half>(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) { return __float2bfloat16(val); }

// Warp Reduce Sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// --- FORWARD KERNEL (Optimized Registers) ---
template<typename scalar_t, int D_MEM>
__global__ void titans_recurrence_fwd_kernel(
    const scalar_t* __restrict__ K, const scalar_t* __restrict__ V, const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ alpha, const scalar_t* __restrict__ theta, const scalar_t* __restrict__ eta,
    const float* __restrict__ M_init, scalar_t* __restrict__ Y,
    int B, int L
) {
    int b = blockIdx.x;
    int i = threadIdx.x; // Thread i handles Row i
    if (b >= B || i >= D_MEM) return;

    // Registers for State (Row i)
    float M_row[D_MEM];
    float S_row[D_MEM];

    // Load Init
    #pragma unroll
    for (int j = 0; j < D_MEM; j++) {
        M_row[j] = M_init[b * D_MEM * D_MEM + i * D_MEM + j];
        S_row[j] = 0.0f; 
    }

    // Shared Memory for Inputs (Small: 3 vectors * 128 * 4B = 1.5KB -> Very Safe)
    __shared__ float s_k[D_MEM];
    __shared__ float s_q[D_MEM];
    
    // Loop over Time
    for (int t = 0; t < L; t++) {
        int base = (b * L + t) * D_MEM;

        // Load K, Q to Shared Mem (Coalesced)
        s_k[i] = to_float(K[base + i]);
        s_q[i] = to_float(Q[base + i]);
        
        // Load scalar params to registers
        float v_val = to_float(V[base + i]);
        float a_t = to_float(alpha[base + i]);
        float th_t = to_float(theta[base + i]);
        float et_t = to_float(eta[base + i]);

        __syncthreads();

        // 1. Compute Error: e = M * k - v
        float dot_Mk = 0.0f;
        #pragma unroll
        for (int j = 0; j < D_MEM; j++) dot_Mk += M_row[j] * s_k[j];
        
        float e_t = dot_Mk - v_val;

        // 2. Update State
        // S = eta * S - theta * (e * k)
        // M = (1-alpha) * M + S
        #pragma unroll
        for (int j = 0; j < D_MEM; j++) {
            float grad = e_t * s_k[j];
            S_row[j] = et_t * S_row[j] - th_t * grad;
            M_row[j] = (1.0f - a_t) * M_row[j] + S_row[j];
        }

        // 3. Output: y = M * q
        float y_val = 0.0f;
        #pragma unroll
        for (int j = 0; j < D_MEM; j++) y_val += M_row[j] * s_q[j];

        Y[base + i] = from_float<scalar_t>(y_val);
        __syncthreads();
    }
}

// --- BACKWARD KERNEL (Re-materialization + Unified Logic) ---
// CHUNK_SIZE nhỏ (16) để giảm áp lực Local Memory trên RTX 3050
template<int D_MEM, int CHUNK_SIZE = 16>
__global__ void titans_recurrence_bwd_kernel_unified(
    const float* __restrict__ K, const float* __restrict__ V, const float* __restrict__ Q,
    const float* __restrict__ alpha, const float* __restrict__ theta, const float* __restrict__ eta,
    const float* __restrict__ M_init, const float* __restrict__ S_init,
    const float* __restrict__ dY,
    float* __restrict__ dK, float* __restrict__ dV, float* __restrict__ dQ,
    float* __restrict__ dalpha, float* __restrict__ dtheta, float* __restrict__ deta,
    float* __restrict__ dM_init, float* __restrict__ dS_init,
    int B, int L
) {
    int b = blockIdx.x;
    int i = threadIdx.x; // Thread i owns Row i
    if (b >= B || i >= D_MEM) return;

    // Pointers
    long batch_offset = (long)b * L * D_MEM;
    long init_offset = (long)b * D_MEM * D_MEM;

    // Registers for Gradients
    float dM_row[D_MEM];
    float dS_row[D_MEM];
    #pragma unroll
    for(int j=0; j<D_MEM; j++) { dM_row[j] = 0.0f; dS_row[j] = 0.0f; }

    // History Buffer in Local Memory (Spills to L1/L2, but safe correctness-wise)
    float M_history[CHUNK_SIZE][D_MEM];
    float S_history[CHUNK_SIZE][D_MEM];
    float e_history[CHUNK_SIZE];

    // Current State Registers
    float M_row[D_MEM];
    float S_row[D_MEM];

    // Shared Memory for exchange
    __shared__ float s_exchange[D_MEM]; 

    int num_chunks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // --- Reverse Loop over Chunks ---
    for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--) {
        int t_start = chunk_idx * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, L);
        int chunk_len = t_end - t_start;

        // 1. RE-MATERIALIZATION (Forward replay from Init to t_end)
        // Reset to Init
        #pragma unroll
        for(int j=0; j<D_MEM; j++) {
            M_row[j] = M_init[init_offset + i * D_MEM + j];
            S_row[j] = S_init[init_offset + i * D_MEM + j];
        }

        // Run forward to fill history
        for (int t = 0; t < t_end; t++) {
            long idx = batch_offset + t * D_MEM;
            
            // Load K to shared for broadcast
            s_exchange[i] = K[idx + i];
            __syncthreads(); 
            // Copy K to registers to free shared mem if needed, but here simple access is OK
            
            float v_val = V[idx + i];
            float a_t = alpha[idx + i];
            float th_t = theta[idx + i];
            float et_t = eta[idx + i];

            // Calc error
            float dot = 0.0f;
            #pragma unroll
            for(int j=0; j<D_MEM; j++) dot += M_row[j] * s_exchange[j]; // M_row * K
            float e_t = dot - v_val;

            // Save history if inside current chunk
            if (t >= t_start) {
                int c = t - t_start;
                e_history[c] = e_t;
                #pragma unroll
                for(int j=0; j<D_MEM; j++) {
                    M_history[c][j] = M_row[j];
                    S_history[c][j] = S_row[j];
                }
            }

            // Update State
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                float grad = e_t * s_exchange[j];
                S_row[j] = et_t * S_row[j] - th_t * grad;
                M_row[j] = (1.0f - a_t) * M_row[j] + S_row[j];
            }
            __syncthreads();
        }

        // 2. BACKWARD PASS (Inside Chunk)
        for (int c = chunk_len - 1; c >= 0; c--) {
            int t = t_start + c;
            long idx = batch_offset + t * D_MEM;

            // Load Inputs
            float val_K = K[idx + i];
            s_exchange[i] = val_K; // Broadcast K
            __syncthreads();
            float local_K[D_MEM];
            #pragma unroll 
            for(int j=0; j<D_MEM; j++) local_K[j] = s_exchange[j];
            __syncthreads();

            // Load Q for dM update
            float val_Q = Q[idx + i];
            s_exchange[i] = val_Q; 
            __syncthreads();
            float local_Q[D_MEM];
            #pragma unroll 
            for(int j=0; j<D_MEM; j++) local_Q[j] = s_exchange[j];
            __syncthreads();

            float val_dy = dY[idx + i];
            float a_t = alpha[idx + i];
            float th_t = theta[idx + i];
            float et_t = eta[idx + i];
            float e_t = e_history[c];

            // Restore M_{t-1}
            float M_prev[D_MEM], S_prev[D_MEM];
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                M_prev[j] = M_history[c][j];
                S_prev[j] = S_history[c][j];
            }

            // --- Gradients ---
            
            // dM_t += dy * q^T
            // Row i of dM accumulates: dy[i] * Q[j]
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                dM_row[j] += val_dy * local_Q[j];
            }

            // dQ (Col reduction): dQ[j] = sum_i (M_t[i,j] * dy[i])
            // Recompute M_t[i,j]
            float M_t_ij[D_MEM];
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                float grad = e_t * local_K[j];
                float S_new = et_t * S_prev[j] - th_t * grad;
                M_t_ij[j] = (1.0f - a_t) * M_prev[j] + S_new;
            }

            // Naive dQ compute: Thread i calculates its contribution to dQ[j]
            // We need to sum across threads.
            // Using smem to reduce is complex here without multiple passes.
            // Fast approximation for 3050: Use Atomic Add to Global since it's cleaner to write,
            // OR use the trick: dQ[i] = sum_k (M_t[k, i] * dy[k]).
            // Let's use AtomicAdd to output dQ/dK to ensure correctness first, 
            // but use registers for intermediate dM/dS.
            // Note: Since we removed the "Inverse" logic, correctness is restored.
            // Using atomicAdd for dK/dQ is acceptable if D is small (64/128).
            
            // dQ accumulation
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                 atomicAdd(&dQ[idx + j], M_t_ij[j] * val_dy);
            }
            
            // Gates and Propagate
            // Total dL/dS_t = dL/dM_t (from M_t = ... + S_t) + future S contribution
            float dS_t[D_MEM];
            #pragma unroll
            for(int j=0; j<D_MEM; j++) dS_t[j] = dM_row[j] + dS_row[j];

            float d_eta = 0.0f, d_theta = 0.0f, de_t = 0.0f;
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                d_eta += dS_t[j] * S_prev[j];
                float G_ij = e_t * local_K[j];
                d_theta -= dS_t[j] * G_ij;
                de_t -= dS_t[j] * th_t * local_K[j];
            }
            
            deta[idx + i] = d_eta;
            dtheta[idx + i] = d_theta;
            dV[idx + i] = -de_t; // dv = -de

            // dK accumulation: dK[j] += M_{t-1}[i,j]*de[i] + ...
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                float term1 = M_prev[j] * de_t;
                float term2 = dS_t[j] * (-th_t * e_t); // from S update
                atomicAdd(&dK[idx + j], term1 + term2);
            }

            // dAlpha
            float d_alpha = 0.0f;
            #pragma unroll
            for(int j=0; j<D_MEM; j++) d_alpha -= dM_row[j] * M_prev[j];
            dalpha[idx + i] = d_alpha;

            // Propagate dM, dS to t-1
            #pragma unroll
            for(int j=0; j<D_MEM; j++) {
                dM_row[j] = (1.0f - a_t) * dM_row[j] + de_t * local_K[j];
                dS_row[j] = et_t * dS_t[j];
            }
            __syncthreads();
        }
    }

    // Save Init Grads
    #pragma unroll
    for (int j = 0; j < D_MEM; j++) {
        dM_init[init_offset + i * D_MEM + j] = dM_row[j];
        dS_init[init_offset + i * D_MEM + j] = dS_row[j];
    }
}

// --- LAUNCHERS ---
#define DISPATCH_DTYPE(dtype_code, ...) \
    if (dtype_code == 1) { using scalar_t = __half; __VA_ARGS__; } \
    else if (dtype_code == 2) { using scalar_t = __nv_bfloat16; __VA_ARGS__; } \
    else { using scalar_t = float; __VA_ARGS__; }

#define DISPATCH_DIM(d_mem, ...) \
    if (d_mem <= 64) { constexpr int D = 64; __VA_ARGS__; } \
    else { constexpr int D = 128; __VA_ARGS__; }

void launch_titans_recurrence_fwd(
    const void* K, const void* V, const void* Q,
    const void* alpha, const void* theta, const void* eta,
    const float* M_init, void* Y,
    int B, int L, int d_mem,
    cudaStream_t stream, int dtype_code
) {
    DISPATCH_DIM(d_mem,
        DISPATCH_DTYPE(dtype_code,
            titans_recurrence_fwd_kernel<scalar_t, D><<<B, D, 0, stream>>>(
                (const scalar_t*)K, (const scalar_t*)V, (const scalar_t*)Q,
                (const scalar_t*)alpha, (const scalar_t*)theta, (const scalar_t*)eta,
                M_init, (scalar_t*)Y, B, L
            );
        )
    )
}

template <typename scalar_t>
void launch_titans_recurrence_bwd(
    const scalar_t* K, const scalar_t* V, const scalar_t* Q,
    const scalar_t* alpha, const scalar_t* theta, const scalar_t* eta,
    const scalar_t* M_init, const scalar_t* S_init,
    const scalar_t* dY,
    scalar_t* dK, scalar_t* dV, scalar_t* dQ,
    scalar_t* dalpha, scalar_t* dtheta, scalar_t* deta,
    scalar_t* dM_init, scalar_t* dS_init,
    int B, int L, int D,
    cudaStream_t stream
) {
    // Luôn dùng kernel Unified (Safe Math)
    // Tự động chọn D=64 hoặc D=128
    if (D <= 64) {
        titans_recurrence_bwd_kernel_unified<64, 16><<<B, 64, 0, stream>>>(
            K, V, Q, alpha, theta, eta, M_init, S_init, dY,
            dK, dV, dQ, dalpha, dtheta, deta, dM_init, dS_init, B, L);
    } else {
        // D=128
        titans_recurrence_bwd_kernel_unified<128, 16><<<B, 128, 0, stream>>>(
            K, V, Q, alpha, theta, eta, M_init, S_init, dY,
            dK, dV, dQ, dalpha, dtheta, deta, dM_init, dS_init, B, L);
    }
}

// Explicit Instantiation
template void launch_titans_recurrence_bwd<float>(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*,
    float*, float*, float*,
    float*, float*, float*,
    float*, float*,
    int, int, int, cudaStream_t);