import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_nvcc_args():
    nvcc_args = [
        '-O3', 
        '--use_fast_math',
        '-allow-unsupported-compiler',
        '-Xcompiler=/permissive-'
    ]
    
    if os.getenv("TORCH_CUDA_ARCH_LIST"):
        print(f"Build with TORCH_CUDA_ARCH_LIST={os.getenv('TORCH_CUDA_ARCH_LIST')}")
        return nvcc_args

    if torch.cuda.is_available():
        arch_list = set()
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            arch_list.add(f"{cap[0]}{cap[1]}")
        
        for arch in arch_list:
            print(f"Detected CUDA Architecture: sm_{arch}")
            nvcc_args.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
            
    else:
        print("Warning: No GPU detected during build. Compiling for common architectures (7.5, 8.0, 8.6, 9.0)...")
        nvcc_args.extend([
            '-gencode=arch=compute_75,code=sm_75', # Turing (RTX 20 series, T4)
            '-gencode=arch=compute_80,code=sm_80', # Ampere (A100)
            '-gencode=arch=compute_86,code=sm_86', # Ampere (RTX 30 series)
            '-gencode=arch=compute_90,code=sm_90', # Hopper (H100)
        ])
        
    return nvcc_args

setup(
    name='mhc_cuda',
    ext_modules=[
        CUDAExtension(
            name='mhc_cuda',
            sources=[
                'src/kernel.cu',
                'src/binding.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': get_nvcc_args()
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)