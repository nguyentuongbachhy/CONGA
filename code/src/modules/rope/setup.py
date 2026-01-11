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
        return nvcc_args

    if torch.cuda.is_available():
        arch_list = set()
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            arch_list.add(f"{cap[0]}{cap[1]}")
        
        for arch in arch_list:
            nvcc_args.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
    else:
        nvcc_args.extend([
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_90,code=sm_90',
        ])
        
    return nvcc_args

setup(
    name='rope_cuda',
    ext_modules=[
        CUDAExtension(
            name='rope_cuda',
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
    cmdclass={'build_ext': BuildExtension}
)
