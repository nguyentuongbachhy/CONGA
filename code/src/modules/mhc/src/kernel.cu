#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define EPSILON 1e-8

template <int N>
__global__ void sinkhorn_fwd_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int n_iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float matrix[N][N];
    int offset = tid * N * N;

    // Load to registers
    #pragma unroll
    for(int i = 0; i < N; ++i) {
        #pragma unroll
        for(int j = 0; j < N; ++j) {
            matrix[i][j] = input[offset + i * N + j];
        }
    }

    // Forward Logic
    for(int iter = 0; iter < n_iters; ++iter) {
        // Row Norm
        #pragma unroll
        for(int i = 0; i < N; ++i) {
            float max_val = -1e9;
            for(int j=0; j<N; ++j) max_val = fmaxf(max_val, matrix[i][j]);
            float sum_exp = 0.0f;
            for(int j=0; j<N; ++j) sum_exp += __expf(matrix[i][j] - max_val);
            float lse = max_val + __logf(sum_exp + EPSILON);
            for(int j=0; j<N; ++j) matrix[i][j] -= lse;
        }
        // Col Norm
        #pragma unroll
        for(int j = 0; j < N; ++j) {
            float max_val = -1e9;
            for(int i=0; i<N; ++i) max_val = fmaxf(max_val, matrix[i][j]);
            float sum_exp = 0.0f;
            for(int i=0; i<N; ++i) sum_exp += __expf(matrix[i][j] - max_val);
            float lse = max_val + __logf(sum_exp + EPSILON);
            for(int i=0; i<N; ++i) matrix[i][j] -= lse;
        }
    }

    // Output exp()
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            output[offset + i * N + j] = __expf(matrix[i][j]);
        }
    }
}

template <int N>
__global__ void sinkhorn_bwd_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ log_alpha,   
    float* __restrict__ grad_input,        
    int batch_size,
    int n_iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    int offset = tid * N * N;

    float matrix[N][N];
    #pragma unroll
    for(int i=0; i<N; ++i) 
        for(int j=0; j<N; ++j) 
            matrix[i][j] = log_alpha[offset + i * N + j];

    for(int iter = 0; iter < n_iters; ++iter) {
        // Row
        #pragma unroll
        for(int i = 0; i < N; ++i) {
            float max_val = -1e9;
            for(int j=0; j<N; ++j) max_val = fmaxf(max_val, matrix[i][j]);
            float sum = 0.0f;
            for(int j=0; j<N; ++j) sum += __expf(matrix[i][j] - max_val);
            float lse = max_val + __logf(sum + EPSILON);
            for(int j=0; j<N; ++j) matrix[i][j] -= lse;
        }
        // Col
        #pragma unroll
        for(int j = 0; j < N; ++j) {
            float max_val = -1e9;
            for(int i=0; i<N; ++i) max_val = fmaxf(max_val, matrix[i][j]);
            float sum = 0.0f;
            for(int i=0; i<N; ++i) sum += __expf(matrix[i][j] - max_val);
            float lse = max_val + __logf(sum + EPSILON);
            for(int i=0; i<N; ++i) matrix[i][j] -= lse;
        }
    }

    float grad[N][N];
    #pragma unroll
    for(int i=0; i<N; ++i) {
        #pragma unroll
        for(int j=0; j<N; ++j) {
            // d(Loss)/d(output) * d(output)/d(matrix)
            // output = exp(matrix) => d(output) = exp(matrix) = output
            // => grad_matrix = grad_output * output
            grad[i][j] = grad_output[offset + i*N + j] * __expf(matrix[i][j]);
        }
    }

    for(int iter = n_iters - 1; iter >= 0; --iter) {
        // 1. Backward Col Norm
        // Jacobian: I - S * 1^T (S là softmax vector của cột)
        #pragma unroll
        for(int j = 0; j < N; ++j) {
            float sum_grad_col = 0.0f;
            for(int i=0; i<N; ++i) sum_grad_col += grad[i][j];
            
            // S_ij = exp(M_ij) / sum_k(exp(M_kj)) = exp(M_ij) 
            // Apply Jacobian: g_new = g_old - S * sum(g_old)
            for(int i=0; i<N; ++i) {
                float prob = __expf(matrix[i][j]); 
                grad[i][j] -= prob * sum_grad_col; 
            }
        }

        // 2. Backward Row Norm
        #pragma unroll
        for(int i = 0; i < N; ++i) {
            float sum_grad_row = 0.0f;
            for(int j=0; j<N; ++j) sum_grad_row += grad[i][j];
            
            // Apply Jacobian: g_new = g_old - S * sum(g_old)
            for(int j=0; j<N; ++j) {
                float prob = __expf(matrix[i][j]);
                grad[i][j] -= prob * sum_grad_row;
            }
        }
    }

    // Write output
    #pragma unroll
    for(int i=0; i<N; ++i) {
        #pragma unroll
        for(int j=0; j<N; ++j) {
            grad_input[offset + i*N + j] = grad[i][j];
        }
    }
}

torch::Tensor mhc_sinkhorn_cuda_forward(torch::Tensor input, int n_iters) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    int n = input.size(-1);
    int numel = input.numel();
    int batch_size = numel / (n * n);
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    if (n == 4) {
        sinkhorn_fwd_kernel<4><<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), batch_size, n_iters);
    } else {
        AT_ERROR("Optimized for n=4 only");
    }
    return output;
}

torch::Tensor mhc_sinkhorn_cuda_backward(torch::Tensor grad_output, torch::Tensor log_alpha, int n_iters) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(log_alpha.is_cuda(), "log_alpha must be CUDA");
    int n = log_alpha.size(-1);
    int numel = log_alpha.numel();
    int batch_size = numel / (n * n);
    auto grad_input = torch::zeros_like(log_alpha);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    if (n == 4) {
        sinkhorn_bwd_kernel<4><<<blocks, threads>>>(
            grad_output.data_ptr<float>(), log_alpha.data_ptr<float>(), grad_input.data_ptr<float>(), batch_size, n_iters);
    } else {
        AT_ERROR("Optimized for n=4 only");
    }
    return grad_input;
}