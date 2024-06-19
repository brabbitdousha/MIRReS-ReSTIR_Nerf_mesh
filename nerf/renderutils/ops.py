# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension
import torch.nn.functional as F

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                vs_editions = glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition) \
                      + glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition)
                paths = sorted(vs_editions, reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/loss.cu',
        'c_src/normal.cu',
        'c_src/common.cpp',
        'c_src/denoising.cu',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import renderutils_plugin
    _cached_plugin = renderutils_plugin
    return _cached_plugin
#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)
NORMAL_THRESHOLD = 0.1

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1)

def _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl):
    smooth_bitang = _safe_normalize(torch.cross(smooth_tng, smooth_nrm))
    if opengl:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] - smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)
    else:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] + smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)
    return _safe_normalize(shading_nrm)

def _bend_normal(view_vec, smooth_nrm, geom_nrm, two_sided_shading):
    # Swap normal direction for backfacing surfaces
    if two_sided_shading:
        smooth_nrm = torch.where(_dot(geom_nrm, view_vec) > 0, smooth_nrm, -smooth_nrm)
        geom_nrm   = torch.where(_dot(geom_nrm, view_vec) > 0, geom_nrm, -geom_nrm)

    t = torch.clamp(_dot(view_vec, smooth_nrm) / NORMAL_THRESHOLD, min=0, max=1)
    return torch.lerp(geom_nrm, smooth_nrm, t)

def _dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)

def bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
    smooth_nrm = _safe_normalize(smooth_nrm)
    smooth_tng = _safe_normalize(smooth_tng)
    view_vec   = _safe_normalize(view_pos - pos)
    shading_nrm = _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl)
    return _bend_normal(view_vec, shading_nrm, geom_nrm, two_sided_shading)

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = _get_plugin().prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return _get_plugin().prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps))

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

class _bilateral_denoiser_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, zdz, sigma):
        ctx.save_for_backward(col, nrm, zdz)
        ctx.sigma = sigma
        out = _get_plugin().bilateral_denoiser_fwd(col, nrm, zdz, sigma)
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        col, nrm, zdz = ctx.saved_variables
        col_grad = _get_plugin().bilateral_denoiser_bwd(col, nrm, zdz, ctx.sigma, out_grad)
        return col_grad, None, None, None

@torch.no_grad()
def bilateral_denoiser_func_no_di(col, nrm, zdz, sigma):
    out = _get_plugin().bilateral_denoiser_fwd(col, nrm, zdz, sigma)
    return out

    
def bilateral_denoiser(h, w, input, factor=1.0):
    input = input.reshape(1, h, w, input.shape[-1])
    sigma = max(factor * 2, 0.0001)
    col    = input[..., 0:3]
    nrm    = safe_normalize(input[..., 3:6]) # Bent normals can produce normals of length < 1 here
    zdz    = input[..., 6:8]
    col_w = _bilateral_denoiser_func.apply(col, nrm, zdz, sigma)
    out_val = col_w[..., 0:3] / col_w[..., 3:4]
    return out_val.view(-1, out_val.shape[-1])

@torch.no_grad()
def bilateral_denoiser_no_di(h, w, input, factor=1.0):
    input = input.reshape(1, h, w, input.shape[-1])
    sigma = max(factor * 2, 0.0001)
    col    = input[..., 0:3]
    nrm    = safe_normalize(input[..., 3:6]) # Bent normals can produce normals of length < 1 here
    zdz    = input[..., 6:8]
    col_w = bilateral_denoiser_func_no_di(col, nrm, zdz, sigma)
    out_val = col_w[..., 0:3] / col_w[..., 3:4]
    return out_val.view(-1, out_val.shape[-1])