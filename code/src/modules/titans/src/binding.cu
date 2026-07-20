#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

void launch_titans_recurrence_fwd(
    const void* K, const void* V, const void* Q,
    const void* alpha, const void* theta, const void* eta,
    const float* M_init, void* Y,
    int B, int L, int d_mem,
    cudaStream_t stream, int dtype_code);

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
    cudaStream_t stream);

static int get_dtype_code(torch::ScalarType dtype) {
    if (dtype == torch::kFloat16) return 1;
    if (dtype == torch::kBFloat16) return 2;
    return 0;
}

torch::Tensor titans_recurrence_forward(
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor Q,
    torch::Tensor alpha,
    torch::Tensor theta,
    torch::Tensor eta,
    torch::Tensor M_init
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dtype = K.scalar_type();
    int dtype_code = get_dtype_code(dtype);

    int B = K.size(0);
    int L = K.size(1);
    int d_mem = K.size(2);

    auto K_c = K.contiguous();
    auto V_c = V.contiguous();
    auto Q_c = Q.contiguous();
    auto alpha_c = alpha.contiguous();
    auto theta_c = theta.contiguous();
    auto eta_c = eta.contiguous();
    auto M_init_f32 = M_init.to(torch::kFloat32).contiguous();

    auto Y = torch::empty({B, L, d_mem}, K.options());

    launch_titans_recurrence_fwd(
        K_c.data_ptr(), V_c.data_ptr(), Q_c.data_ptr(),
        alpha_c.data_ptr(), theta_c.data_ptr(), eta_c.data_ptr(),
        M_init_f32.data_ptr<float>(), Y.data_ptr(),
        B, L, d_mem, stream, dtype_code
    );

    return Y;
}

std::vector<torch::Tensor> titans_recurrence_backward(
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor Q,
    torch::Tensor alpha,
    torch::Tensor theta,
    torch::Tensor eta,
    torch::Tensor M_init,
    torch::Tensor S_init,
    torch::Tensor dY
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    
    int B = K.size(0);
    int L = K.size(1);
    int d_mem = K.size(2);
    
    auto K_c = K.contiguous();
    auto V_c = V.contiguous();
    auto Q_c = Q.contiguous();
    auto alpha_c = alpha.contiguous();
    auto theta_c = theta.contiguous();
    auto eta_c = eta.contiguous();
    auto M_init_c = M_init.contiguous();
    auto S_init_c = S_init.contiguous();
    auto dY_c = dY.contiguous();
    
    auto dK = torch::zeros_like(K_c);
    auto dV = torch::zeros_like(V_c);
    auto dQ = torch::zeros_like(Q_c);
    auto dalpha = torch::zeros_like(alpha_c);
    auto dtheta = torch::zeros_like(theta_c);
    auto deta = torch::zeros_like(eta_c);
    auto dM_init = torch::zeros_like(M_init_c);
    auto dS_init = torch::zeros_like(S_init_c);
    
    auto K_f = K_c.to(torch::kFloat32).contiguous();
    auto V_f = V_c.to(torch::kFloat32).contiguous();
    auto Q_f = Q_c.to(torch::kFloat32).contiguous();
    auto alpha_f = alpha_c.to(torch::kFloat32).contiguous();
    auto theta_f = theta_c.to(torch::kFloat32).contiguous();
    auto eta_f = eta_c.to(torch::kFloat32).contiguous();
    auto M_init_f = M_init_c.to(torch::kFloat32).contiguous();
    auto S_init_f = S_init_c.to(torch::kFloat32).contiguous();
    auto dY_f = dY_c.to(torch::kFloat32).contiguous();
    
    auto dK_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dV_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dQ_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dalpha_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dtheta_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto deta_f = torch::zeros({B, L, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dM_init_f = torch::zeros({B, d_mem, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    auto dS_init_f = torch::zeros({B, d_mem, d_mem}, torch::TensorOptions().dtype(torch::kFloat32).device(K.device()));
    
    launch_titans_recurrence_bwd<float>(
        K_f.data_ptr<float>(), V_f.data_ptr<float>(), Q_f.data_ptr<float>(),
        alpha_f.data_ptr<float>(), theta_f.data_ptr<float>(), eta_f.data_ptr<float>(),
        M_init_f.data_ptr<float>(), S_init_f.data_ptr<float>(),
        dY_f.data_ptr<float>(),
        dK_f.data_ptr<float>(), dV_f.data_ptr<float>(), dQ_f.data_ptr<float>(),
        dalpha_f.data_ptr<float>(), dtheta_f.data_ptr<float>(), deta_f.data_ptr<float>(),
        dM_init_f.data_ptr<float>(), dS_init_f.data_ptr<float>(),
        B, L, d_mem, stream
    );
    
    dK = dK_f.to(K.scalar_type());
    dV = dV_f.to(V.scalar_type());
    dQ = dQ_f.to(Q.scalar_type());
    dalpha = dalpha_f.to(alpha.scalar_type());
    dtheta = dtheta_f.to(theta.scalar_type());
    deta = deta_f.to(eta.scalar_type());
    dM_init = dM_init_f.to(M_init.scalar_type());
    dS_init = dS_init_f.to(S_init.scalar_type());
    
    return {dK, dV, dQ, dalpha, dtheta, deta, dM_init, dS_init};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recurrence_forward", &titans_recurrence_forward, "TITANS Recurrence Forward (CUDA)");
    m.def("recurrence_backward", &titans_recurrence_backward, "TITANS Recurrence Backward (CUDA)");
}