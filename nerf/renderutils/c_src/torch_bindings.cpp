// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifdef _MSC_VER 
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define NVDR_CHECK_GL_ERROR(GL_CALL) { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); }
#define CHECK_TENSOR(X, DIMS, CHANNELS) \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor") \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions") \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "loss.h"
#include "normal.h"
#include "denoising.h"
#include "common_de.h"

#define BLOCK_X 8
#define BLOCK_Y 8

// CUDA kernels

void bilateral_denoiser_fwd_kernel(BilateralDenoiserParams params);
void bilateral_denoiser_bwd_kernel(BilateralDenoiserParams params);

//------------------------------------------------------------------------
// loss.cu

void imgLossFwdKernel(LossKernelParams p);
void imgLossBwdKernel(LossKernelParams p);

//------------------------------------------------------------------------
// normal.cu

void PrepareShadingNormalFwdKernel(PrepareShadingNormalKernelParams p);
void PrepareShadingNormalBwdKernel(PrepareShadingNormalKernelParams p);

//------------------------------------------------------------------------
// bsdf.cu

//------------------------------------------------------------------------
// Tensor helpers

void update_grid(dim3 &gridSize, torch::Tensor x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
}

template<typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
    update_grid(gridSize, std::forward<Ts>(vs)...);
}

Tensor make_cuda_tensor(torch::Tensor val)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = 1; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::empty({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));
        else // 3
            *grad = torch::empty({ outDims.z, outDims.x, val.size(2) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}

//------------------------------------------------------------------------
// prepare_shading_normal

torch::Tensor prepare_shading_normal_fwd(torch::Tensor pos, torch::Tensor view_pos, torch::Tensor perturbed_nrm, torch::Tensor smooth_nrm, torch::Tensor smooth_tng, torch::Tensor geom_nrm, bool two_sided_shading, bool opengl, bool fp16)
{
    CHECK_TENSOR(pos, 4, 3);
    CHECK_TENSOR(view_pos, 4, 3);
    CHECK_TENSOR(perturbed_nrm, 4, 3);
    CHECK_TENSOR(smooth_nrm, 4, 3);
    CHECK_TENSOR(smooth_tng, 4, 3);
    CHECK_TENSOR(geom_nrm, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PrepareShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.pos = make_cuda_tensor(pos, p.gridSize);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PrepareShadingNormalFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> prepare_shading_normal_bwd(torch::Tensor pos, torch::Tensor view_pos, torch::Tensor perturbed_nrm, torch::Tensor smooth_nrm, torch::Tensor smooth_tng, torch::Tensor geom_nrm, torch::Tensor grad, bool two_sided_shading, bool opengl)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PrepareShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad;
    p.pos = make_cuda_tensor(pos, p.gridSize, &pos_grad);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize, &view_pos_grad);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize, &perturbed_nrm_grad);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize, &smooth_nrm_grad);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize, &smooth_tng_grad);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize, &geom_nrm_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PrepareShadingNormalBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad);
}

template<class T, int N, template <typename U> class PtrTraits = DefaultPtrTraits> PackedTensorAccessor32<T, N> packed_accessor32(torch::Tensor tensor)
{
    return PackedTensorAccessor32<T,N,PtrTraits>(static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()), tensor.sizes().data(), tensor.strides().data());
}

torch::Tensor bilateral_denoiser_fwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor zdz, float sigma)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({ col.size(0), col.size(1), col.size(2), 4 }, opts);

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((col.size(2) - 1) / blockSize.x + 1, (col.size(1) - 1) / blockSize.y + 1, (col.size(0) - 1) / blockSize.z + 1);

    BilateralDenoiserParams params;
    params.col = packed_accessor32<float, 4>(col);
    params.nrm = packed_accessor32<float, 4>(nrm);
    params.zdz = packed_accessor32<float, 4>(zdz);
    params.out = packed_accessor32<float, 4>(out);
    params.sigma = sigma;

    void *args[] = {&params};
    CUDA_CHECK(cudaLaunchKernel((const void *)bilateral_denoiser_fwd_kernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor bilateral_denoiser_bwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor zdz, float sigma, torch::Tensor out_grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor col_grad = torch::zeros({ col.size(0), col.size(1), col.size(2), col.size(3) }, opts);

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((col.size(2) - 1) / blockSize.x + 1, (col.size(1) - 1) / blockSize.y + 1, (col.size(0) - 1) / blockSize.z + 1);

    BilateralDenoiserParams params;
    params.col = packed_accessor32<float, 4>(col);
    params.nrm = packed_accessor32<float, 4>(nrm);
    params.zdz = packed_accessor32<float, 4>(zdz);
    params.out_grad = packed_accessor32<float, 4>(out_grad);
    params.col_grad = packed_accessor32<float, 4>(col_grad);
    params.sigma = sigma;

    void *args[] = {&params};
    CUDA_CHECK(cudaLaunchKernel((const void *)bilateral_denoiser_bwd_kernel, gridSize, blockSize, args, 0, stream));

    return col_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_shading_normal_fwd", &prepare_shading_normal_fwd, "prepare_shading_normal_fwd");
    m.def("prepare_shading_normal_bwd", &prepare_shading_normal_bwd, "prepare_shading_normal_bwd");
    m.def("bilateral_denoiser_fwd", &bilateral_denoiser_fwd, "bilateral_denoiser_fwd");
    m.def("bilateral_denoiser_bwd", &bilateral_denoiser_bwd, "bilateral_denoiser_bwd");
}