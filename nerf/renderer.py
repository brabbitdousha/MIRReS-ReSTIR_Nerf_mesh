import os
import cv2
import math
import json
import tqdm
import mcubes
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
import nvdiffrast.torch as dr

import xatlas
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

from .utils import custom_meshgrid, plot_pointcloud, safe_normalize, act_voffsets, linear2srgb_torch, pixel_grid, srgb_to_linear, linear_to_srgb_np
from meshutils import *
from .renderutils.ops import prepare_shading_normal
from .render_dump import dump_render
from .renderer_restir import *
import pyexr

lock_fn_triro = os.path.join(torch.utils.cpp_extension._get_build_directory('triro', False), 'lock')
if os.path.exists(lock_fn_triro):
    print("Warning: Lock file exists in build directory: '%s'" % lock_fn_triro)

#from triro.ray.ray_optix import RayMeshIntersector

@torch.no_grad()
def generate_light_image(base, res):
        texcoord = pixel_grid(res[1], res[0])
        return dr.texture(base[None, ...].contiguous(), texcoord[None, ...].contiguous(), filter_mode='linear')[0]

def contract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    return xyzs

def uncontract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
    return xyzs

# import torch_scatter
TORCH_SCATTER = None # lazy import

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = 2
        else:
            self.bound = opt.bound
        
        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.grid_size = opt.grid_size
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        self.max_level = 16

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # individual codes
        self.individual_num = opt.ind_num
        self.individual_dim = opt.ind_dim

        #if self.individual_dim > 0:
        #    self.individual_codes = nn.Parameter(torch.randn(self.individual_num, self.individual_dim) * 0.1) 
        #else:
        self.individual_codes = None #not used for ngp nerf

        # extra state for cuda raymarching
        self.cuda_ray = opt.cuda_ray
        assert self.cuda_ray

        # density grid
        if not self.opt.trainable_density_grid:
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            self.register_buffer('density_grid', density_grid)
        else:
            self.density_grid = nn.Parameter(torch.zeros([self.cascade, self.grid_size ** 3])) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0
        
        # for second phase training

        if self.opt.stage == 1:
            
            if self.opt.gui:
                self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            else:
                self.glctx = dr.RasterizeGLContext(output_db=False) # will crash if using GUI...

            # sequentially load cascaded meshes
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            for cas in range(self.cascade):
                
                _updated_mesh_path = os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}_updated.ply') if self.opt.mesh == '' else self.opt.mesh
                if os.path.exists(_updated_mesh_path) and self.opt.ckpt != 'scratch':
                    mesh = trimesh.load(_updated_mesh_path, force='mesh', skip_material=True, process=False)
                else: # base (not updated)
                    mesh = trimesh.load(os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}.ply'), force='mesh', skip_material=True, process=False)
                print(f'[INFO] loaded cascade {cas} mesh: {mesh.vertices.shape}, {mesh.faces.shape}')

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

            vertices = np.concatenate(vertices, axis=0)
            triangles = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

            # must put to cuda manually, we don't want these things in the model as buffers...
            self.vertices = torch.from_numpy(vertices).float().cuda() # [N, 3]
            self.triangles = torch.from_numpy(triangles).int().cuda()

            # learnable offsets for mesh vertex
            self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

            if self.opt.use_brdf:
                if not self.opt.use_restir:
                    self.intersector = None#RayMeshIntersector(self.vertices.detach(), self.triangles.detach())
                else:
                    self.bvh_restir_worker = restirbvhWorker(self.vertices.detach(), self.triangles.detach())

            # accumulate error for mesh face
            self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_id = None
            
        else:
            self.glctx = None

    def get_params(self, lr):
        params = []

        if self.individual_codes is not None:
            params.append({'params': self.individual_codes, 'lr': self.opt.lr * 0.1, 'weight_decay': 0})

        if self.opt.trainable_density_grid:
            params.append({'params': self.density_grid, 'lr': self.opt.lr, 'weight_decay': 0})

        if self.glctx is not None:
            params.append({'params': self.vertices_offsets, 'lr': self.opt.lr_vert, 'weight_decay': 0})

        return params
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    def update_aabb(self, aabb):
        # aabb: tensor of [6]
        if not torch.is_tensor(aabb):
            aabb = torch.from_numpy(aabb).float()
        self.aabb_train = aabb.clamp(-self.real_bound, self.real_bound).to(self.aabb_train.device)
        self.aabb_infer = self.aabb_train.clone()
        print(f'[INFO] update_aabb: {self.aabb_train.cpu().numpy().tolist()}')

    @torch.no_grad()
    def refine_and_decimate(self):

        assert self.opt.stage > 0
        device = self.vertices.device
        voff = act_voffsets(self.vertices_offsets)
        v = (self.vertices + voff).detach().cpu().numpy()
        f = self.triangles.detach().cpu().numpy()

        errors = self.triangles_errors.cpu().numpy()

        cnt = self.triangles_errors_cnt.cpu().numpy()
        cnt_mask = cnt > 0
        errors[cnt_mask] = errors[cnt_mask] / cnt[cnt_mask]

        # only care about the inner mesh
        errors = errors[:self.f_cumsum[1]]
        cnt_mask = cnt_mask[:self.f_cumsum[1]]

        if self.opt.sdf:

            # sdf mode: just set all faces to decimation & remesh.
            mask = np.ones_like(errors)

        else:

            # find a threshold to decide whether we perform subdivision / decimation.
            thresh_refine = np.percentile(errors[cnt_mask], 90)
            thresh_decimate = np.percentile(errors[cnt_mask], 50)

            mask = np.zeros_like(errors)
            mask[(errors > thresh_refine) & cnt_mask] = 2
            mask[(errors < thresh_decimate) & cnt_mask] = 1

            print(f'[INFO] faces to decimate {(mask == 1).sum()}, faces to refine {(mask == 2).sum()}')

        if self.bound <= 1:

            # mesh = trimesh.Trimesh(v, f, process=False)
            # mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_before_updated.ply'))

            v, f = decimate_and_refine_mesh(v, f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)
            # export
            mesh = trimesh.Trimesh(v, f, process=False)
            mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_updated.ply'))
            v, f = mesh.vertices, mesh.faces

            # fix counters
            self.v_cumsum[1:] += v.shape[0] - self.v_cumsum[1]
            self.f_cumsum[1:] += f.shape[0] - self.f_cumsum[1]
        else:
            
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]

            for cas in range(self.cascade):

                cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
                cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]

                if cas == 0:
                    cur_v, cur_f = decimate_and_refine_mesh(cur_v, cur_f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)

                mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
                mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}_updated.ply'))

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
            
            v = np.concatenate(vertices, axis=0)
            f = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

        self.vertices = torch.from_numpy(v).float().contiguous().to(device) # [N, 3]
        self.triangles = torch.from_numpy(f).int().contiguous().to(device)
        self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

        self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)

        print(f'[INFO] update stage1 mesh: {self.vertices.shape}, {self.triangles.shape}')


    @torch.no_grad()
    def export_stage1(self, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)

        assert self.opt.stage > 0
        device = self.vertices.device

        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor

            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uv in contracted space
            atlas = xatlas.Atlas()
            atlas.add_mesh(contract(v_np) if self.opt.contract else v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0 # disable merge_chart for faster unwrap...
            pack_options = xatlas.PackOptions()
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            if self.opt.contract:
                xyzs = contract(xyzs)
            
            feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # check individual codes
                if self.individual_dim > 0:
                    ind_code = self.individual_codes[[0]]
                else:
                    ind_code = None

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        all_feats.append(self.mlp_mat_opt.sample(xyzs[head:tail]).float())#all_feats.append(self.geo_feat(xyzs[head:tail], ind_code).float())
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1) # 6 channels
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = np.clip(feats, 0, 1)
            feats = linear_to_srgb_np(feats)
            feats = (feats * 255).astype(np.uint8)

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=32) # pad width
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR) # albedo
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR) # visibility features

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            # cv2.imwrite(os.path.join(path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            # cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
            cv2.imwrite(os.path.join(path, f'feat1_{cas}.jpg'), feats1)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'mesh_{cas}.obj')
            mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:

                fp.write(f'mtllib mesh_{cas}.mtl \n')
                
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                fp.write(f'map_Kd feat0_{cas}.jpg \n')
                
        voff = act_voffsets(self.vertices_offsets)
        v = (self.vertices + voff).detach()
        f = self.triangles.detach()

        for cas in range(self.cascade):
            cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
            cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]
            _export_obj(cur_v, cur_f, h0, w0, self.opt.ssaa, cas)

            # half the texture resolution for remote area.
            if not self.opt.sdf and h0 > 2048 and w0 > 2048:
                h0 //= 2
                w0 //= 2

        # save mlp as json
        '''
        params = dict(self.specular_net.named_parameters())

        mlp = {}
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f'[INFO] wrting MLP param {k}: {p_np.shape}')
            mlp[k] = p_np.tolist()

        mlp['bound'] = self.bound
        mlp['cascade'] = self.cascade

        mlp_file = os.path.join(path, f'mlp.json')
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)
        '''

    
    @torch.no_grad()
    def export_stage0(self, save_path, resolution=None, decimate_target=1e5, dataset=None, S=128):

        # only for the inner mesh inside [-1, 1]
        if resolution is None:
            resolution = self.grid_size

        device = self.density_bitfield.device

        density_thresh = min(self.mean_density, self.density_thresh)

        # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
        sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        if resolution == self.grid_size:
            # re-map from morton code to regular coords...
            all_indices = torch.arange(resolution**3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            sigmas[tuple(all_coords.T)] = self.density_grid[0]
        else:
            # query
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                            val = self.density(pts.to(device))['sigma'] # [S, 1]
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 

            # use the density_grid as a baseline mask (also excluding untrained regions)
            if not self.opt.sdf:
                mask = torch.zeros([self.grid_size] * 3, dtype=torch.float32, device=device)
                all_indices = torch.arange(self.grid_size**3, device=device, dtype=torch.int)
                all_coords = raymarching.morton3D_invert(all_indices).long()
                mask[tuple(all_coords.T)] = self.density_grid[0]
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[resolution] * 3, mode='nearest').squeeze(0).squeeze(0)
                mask = (mask > density_thresh)
                sigmas = sigmas * mask

        sigmas = torch.nan_to_num(sigmas, 0)
        sigmas = sigmas.cpu().numpy()

        # import kiui
        # for i in range(254,255):
        #     kiui.vis.plot_matrix((sigmas[..., i]).astype(np.float32))

        if self.opt.sdf:
            vertices, triangles = mcubes.marching_cubes(-sigmas, 0)
        else:
            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)

        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)

        ### visibility test.
        if dataset is not None:
            visibility_mask = self.mark_unseen_triangles(vertices, triangles, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
            vertices, triangles = remove_masked_trigs(vertices, triangles, visibility_mask, dilation=self.opt.visibility_mask_dilation)

        ### reduce floaters by post-processing...
        vertices, triangles = clean_mesh(vertices, triangles, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=True, remesh=False)
        
        ### decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, remesh=False)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(os.path.join(save_path, f'mesh_0.ply'))

        # for the outer mesh [1, inf]
        if self.bound > 1:
            
            if self.opt.sdf:
                # assume background contracted in [-2, 2], process it specially
                sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            pts = 2 * torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                val = self.density(pts.to(device))['sigma'] # [S, 1]
                            sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
                sigmas = torch.nan_to_num(sigmas, 0)
                sigmas = sigmas.cpu().numpy()

                vertices_out, triangles_out = mcubes.marching_cubes(-sigmas, 0)

                vertices_out = vertices_out / (resolution - 1.0) * 2 - 1
                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)

                _r = 0.5
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')

                bound = 2
                half_grid_size = bound / resolution

                vertices_out = vertices_out * (bound - half_grid_size)

                # clean mesh
                vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)

                # decimate
                decimate_target *= 2
                if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                    vertices_out, triangles_out = decimate_mesh(vertices_out, triangles_out, decimate_target, optimalplacement=False)

                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)

                # warp back (uncontract)
                vertices_out = uncontract(vertices_out)

                # remove the out-of-AABB region
                xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})')

                if dataset is not None:
                    visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
                    vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask, dilation=self.opt.visibility_mask_dilation)

                print(f'[INFO] exporting outer mesh at cas 1, v = {vertices_out.shape}, f = {triangles_out.shape}')
                
                # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                mesh_out = trimesh.Trimesh(vertices_out, triangles_out, process=False) # important, process=True leads to seg fault...
                mesh_out.export(os.path.join(save_path, f'mesh_1.ply'))

                
            else:
                reso = self.grid_size
                target_reso = self.opt.env_reso
                decimate_target //= 2 # empirical...

                all_indices = torch.arange(reso**3, device=device, dtype=torch.int)
                all_coords = raymarching.morton3D_invert(all_indices).cpu().numpy()

                # for each cas >= 1
                for cas in range(1, self.cascade):
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / target_reso

                    # remap from density_grid
                    occ = torch.zeros([reso] * 3, dtype=torch.float32, device=device)
                    occ[tuple(all_coords.T)] = self.density_grid[cas]

                    # remove the center (before mcubes)
                    # occ[reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4] = 0

                    # interpolate the occ grid to desired resolution to control mesh size...
                    occ = F.interpolate(occ.unsqueeze(0).unsqueeze(0), [target_reso] * 3, mode='trilinear').squeeze(0).squeeze(0)
                    occ = torch.nan_to_num(occ, 0)
                    occ = (occ > density_thresh).cpu().numpy()

                    vertices_out, triangles_out = mcubes.marching_cubes(occ, 0.5)

                    vertices_out = vertices_out / (target_reso - 1.0) * 2 - 1 # range in [-1, 1]

                    # remove the center (already covered by previous cascades)
                    _r = 0.45
                    vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')
                    if vertices_out.shape[0] == 0: continue

                    vertices_out = vertices_out * (bound - half_grid_size)

                    # remove the out-of-AABB region
                    xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
                    xmn += half_grid_size
                    ymn += half_grid_size
                    zmn += half_grid_size
                    xmx -= half_grid_size
                    ymx -= half_grid_size
                    zmx -= half_grid_size
                    vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})')

                    # clean mesh
                    vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)

                    if vertices_out.shape[0] == 0: continue

                    # decimate
                    if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                        vertices_out, triangles_out = decimate_mesh(vertices_out, triangles_out, decimate_target, optimalplacement=False)

                    vertices_out = vertices_out.astype(np.float32)
                    triangles_out = triangles_out.astype(np.int32)

                    print(f'[INFO] exporting outer mesh at cas {cas}, v = {vertices_out.shape}, f = {triangles_out.shape}')
            
                    if dataset is not None:
                        visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
                        vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask, dilation=self.opt.visibility_mask_dilation)
                    
                    # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                    mesh_out = trimesh.Trimesh(vertices_out, triangles_out, process=False) # important, process=True leads to seg fault...
                    mesh_out.export(os.path.join(save_path, f'mesh_{cas}.ply'))


    # phase 0 continuous training
    def render(self, rays_o, rays_d, index=None, dt_gamma=0, bg_color=None, perturb=False, max_steps=1024, T_thresh=1e-4, cam_near_far=None, shading='full', **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        # print((nears < fars).sum(), nears.shape[0])
        
        # mix background color
        if bg_color is None:
            bg_color = 1
        
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[[0]]
        else:
            ind_code = None

        results = {}

        if self.training:
            
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)

            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            # ray-wise to point-wise
            if ind_code is not None and ind_code.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                ind_code = ind_code[flatten_rays]

            dirs = safe_normalize(dirs)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                sigmas, rgbs, speculars = self(xyzs, dirs, ind_code, shading)
            
            if self.opt.sdf:
                # sigmas are sdf
                inv_s = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)
                raw_normal = self.normal(xyzs, self.opt.normal_anneal_epsilon)
                results['normal'] = raw_normal
                normal = safe_normalize(raw_normal)
                true_cos = (dirs * normal).sum(-1)
                iter_cos = - (F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.opt.cos_anneal_ratio) + \
                              F.relu(-true_cos) * self.opt.cos_anneal_ratio)
                estimated_prev_sdf = sigmas - iter_cos * ts[:, 1] * 0.5
                estimated_next_sdf = sigmas + iter_cos * ts[:, 1] * 0.5
                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                p = prev_cdf - next_cdf
                c = prev_cdf
                sigmas = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0, 1) # sigmas are alpha now

            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, self.opt.sdf)

            results['num_points'] = xyzs.shape[0]
            results['xyzs'] = xyzs
            results['speculars'] = speculars
            results['weights'] = weights
            results['weights_sum'] = weights_sum

        else:
            # allocate outputs 
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, dt_gamma, max_steps)
                
                dirs = safe_normalize(dirs)
                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    sigmas, rgbs, speculars = self(xyzs, dirs, ind_code, shading)
                
                if self.opt.sdf:
                    # sigmas are sdf
                    inv_s = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)
                    raw_normal = self.normal(xyzs)
                    normal = safe_normalize(raw_normal)
                    true_cos = - F.relu(-(dirs * normal).sum(-1))
                    estimated_prev_sdf = sigmas - true_cos * ts[:, 1] * 0.5
                    estimated_next_sdf = sigmas + true_cos * ts[:, 1] * 0.5
                    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
                    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
                    p = prev_cdf - next_cdf
                    c = prev_cdf
                    sigmas = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0, 1)

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, self.opt.sdf)

                rays_alive = rays_alive[rays_alive >= 0]

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        results['depth'] = depth
        results['image'] = image

        return results
    
    @torch.no_grad()
    def compute_secondary_shading_effects(self, surface_pts, surf2light, chunk_size, vis_near, vis_far, index=None, dt_gamma=0, bg_color=None, perturb=False, max_steps=1024, T_thresh=1e-4, cam_near_far=None, shading='full'
                                            ):
        '''compute visibility for each point at each direction without visbility network
        - args:
            - tensoIR: tensoIR model is used to compute the visibility and indirect lighting
            - surface_pts: [N, 3] surface points location
            - surf2light: [N, 3], light incident direction for each surface point, pointing from surface to light
            - light_idx: [N, 1], index of lighitng
            - nSample: number of samples for each ray along incident light direction
        - return:
            - visibility_compute: [N, 1] visibility result by choosing some directions and then computing the density
            - indirect_light: [N, 3] indirect light in the corresponding direction
        '''
        device = surface_pts.device
        visibility_compute = torch.zeros((surface_pts.shape[0]), dtype=torch.float32).to(device) # [N, 1]
        indirect_light = torch.zeros((surface_pts.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]

        chunk_idxs_vis_compute = torch.split(torch.arange(surface_pts.shape[0]), chunk_size)
        for chunk_idx in chunk_idxs_vis_compute:
            chunk_surface_pts = surface_pts[chunk_idx]  # [chunk_size, 3]
            chunk_surf2light = surf2light[chunk_idx]    # [chunk_size, 3]
            vis_chunk, indirect_light_chunk = self.compute_vis_and_radiance(
                                                                chunk_surface_pts
                                                                ,chunk_surf2light
                                                                ,vis_near, vis_far
                                                                ,index=index, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far, shading=shading
                                                                ) 


            visibility_chunk = vis_chunk
            visibility_compute[chunk_idx] = visibility_chunk
            indirect_light[chunk_idx] = indirect_light_chunk


        visibility_compute = visibility_compute.reshape(-1, 1)  # [N, 1]
        indirect_light = indirect_light.reshape(-1, 3)  # [N, 3]

        return visibility_compute, indirect_light
    
    def compute_vis_and_radiance(self, rays_o, rays_d, vis_near, vis_far, index=None, dt_gamma=0, bg_color=None, perturb=False, max_steps=1024, T_thresh=1e-4, cam_near_far=None, shading='full'):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        nears = torch.maximum(nears, torch.tensor(vis_near))
        fars = torch.minimum(fars, torch.tensor(vis_far))
        # print((nears < fars).sum(), nears.shape[0])
        
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[[0]]
        else:
            ind_code = None
           
        xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # ray-wise to point-wise
        if ind_code is not None and ind_code.shape[0] > 1:
            flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
            ind_code = ind_code[flatten_rays]

        dirs = safe_normalize(dirs)
        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
            sigmas, rgbs, speculars = self(xyzs, dirs, ind_code, shading)

        weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, self.opt.sdf)

        image = image.view(*prefix, 3)

        return 1 - weights_sum, image
    
    # phase 2
    def render_stage1(self, rays_o, rays_d, mvp, h0, w0, index=None, bg_color=None, shading='full', is_test=False, **kwargs):

        prefix = rays_d.shape[:-1]
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_d.shape[0] # N = B * N, in fact
        device = rays_d.device
            
        # do super-sampling
        if self.opt.ssaa > 1:
            h = int(h0 * self.opt.ssaa)
            w = int(w0 * self.opt.ssaa)
            # interpolate rays_d when ssaa > 1 ...
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w = h0, w0
            dirs = rays_d.contiguous()

        dirs = safe_normalize(dirs)

        # mix background color
        if bg_color is None:
            bg_color = 1

        # [N, 3] to [h, w, 3]
        if torch.is_tensor(bg_color) and len(bg_color.shape) == 2:
            bg_color = bg_color.view(h0, w0, 3)
        
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[[0]]
        else:
            ind_code = None

        results = {}

        voff = act_voffsets(self.vertices_offsets)
        vertices = self.vertices + voff # [N, 3]
        if self.opt.use_brdf:
            if not self.opt.use_restir:
                self.intersector.update_raw(vertices.detach(), self.triangles.detach())
            else:
                #vert_de = vertices.detach()
                #vert_ind_de = self.triangles.detach()
                self.bvh_restir_worker.update_mesh(vertices.detach(), self.triangles.detach())
                #LBVHNode_info, LBVHNode_aabb = get_bvh(vert_de, vert_ind_de, self.m_gen_ele, self.m_morton_codes, self.m_radixsort, self.m_hierarchy, self.m_bounding_box)

            #compute normal
            v_nrm, t_nrm_idx = auto_normals(vertices, self.triangles)

        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4]

        rast, rast_out_deriv_s = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))

        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles) # [1, H, W, 3]

        if self.opt.use_brdf:
            #get gbuffer normal 
            perturbed_nrm = None
            # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
            v0 = vertices[self.triangles[:, 0].long(), :]
            v1 = vertices[self.triangles[:, 1].long(), :]
            v2 = vertices[self.triangles[:, 2].long(), :]
            face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0))
            face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
            gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast, face_normal_indices.int())

            gb_normal, _ = dr.interpolate(v_nrm[None, ...], rast, t_nrm_idx.int())

            #jitter
            offset = torch.normal(mean=0, std=0.005, size=(xyzs.shape[0], xyzs.shape[1], xyzs.shape[2], 2), device="cuda")
            jitter = (pixel_grid(xyzs.shape[2], xyzs.shape[1])[None, ...] + offset).contiguous()

            mask = (rast[..., -1:] > 0).float()
            mask_tap = dr.texture(mask.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
            grad_weight = mask * mask_tap

            # Geometric smoothed normal regularizer
            nrm_jitter = dr.texture(gb_normal.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
            nrm_grad = torch.abs(nrm_jitter - gb_normal) * grad_weight

            gb_tangent = torch.zeros_like(gb_normal)
            gb_normal = prepare_shading_normal(xyzs, rays_o[0,:].view(1,1,1,3), perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)
            gb_normal = safe_normalize(gb_normal)

            #get kd ks
            all_tex_jitter = self.mlp_mat_opt.sample(xyzs + torch.normal(mean=0, std=0.01, size=xyzs.shape, device="cuda"))
            all_tex = self.mlp_mat_opt.sample(xyzs)
            assert all_tex.shape[-1] == 6, "Combined kd_ks must be 6 channels"
            kd, ks = all_tex[..., 0:3], all_tex[..., 3:6]
            kd_grad  = torch.abs(all_tex_jitter[..., 0:3] - kd)
            ks_grad  = torch.abs(all_tex_jitter[..., 3:6] - ks) * torch.tensor([0, 1, 1], dtype=torch.float32, device='cuda')[None, None, None, :] # Omit o-component
            #get envmap
            env_map = self.lgt.base
            if self.opt.test and self.opt.envmap_path != 'None':
                env_map = self.env_map0

        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles) # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1).detach()
        xyzs = xyzs.view(-1, 3)

        # random noise to make appearance more robust
        # if self.training:
        #     xyzs = xyzs + torch.randn_like(xyzs) * 1e-3
        if self.opt.contract:
            xyzs = contract(xyzs)

        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        brdf_rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        brdf_diff = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        brdf_spec = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        rgb_diff = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        rgb_spec = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                mask_rgbs, masked_specular = self.rgb(
                    xyzs[mask_flatten] if self.opt.enable_offset_nerf_grad else xyzs[mask_flatten].detach(), 
                    dirs[mask_flatten], ind_code, shading)

            rgbs[mask_flatten] = mask_rgbs.float()

            if self.opt.use_brdf:
                rgb_diff[mask_flatten] = mask_rgbs.detach().float()
                rgb_spec[mask_flatten] = masked_specular.detach().float()
                #brdf rendering

                #debug space----------------------
                #kd = torch.ones_like(xyzs)
                #roughness_map = torch.full(kd.shape, 0.8).cuda()
                #fresnel_map = torch.full(kd.shape, 0.04).cuda()
                #env_map = torch.zeros((envmap_h, envmap_w,3),device=device, dtype=torch.float32)
                #env_map[envmap_h//4, envmap_w//2, :] = 1
                #env_map.reshape(-1 , 3)
                #---------------------------------
                if self.opt.use_restir:

                    # Interpolate z and z-gradient
                    gb_depth = None
                    if self.opt.use_bi_de:
                        with torch.no_grad():
                            eps = 0.00001
                            clip_pos, clip_pos_deriv = dr.interpolate(vertices_clip, rast, self.triangles, rast_db=rast_out_deriv_s)
                            z0 = torch.clamp(clip_pos[..., 2:3], min=eps) / torch.clamp(clip_pos[..., 3:4], min=eps)
                            z1 = torch.clamp(clip_pos[..., 2:3] + torch.abs(clip_pos_deriv[..., 2:3]), min=eps) / torch.clamp(clip_pos[..., 3:4] + torch.abs(clip_pos_deriv[..., 3:4]), min=eps)
                            z_grad = torch.abs(z1 - z0)
                            if torch.numel(z_grad) == 0:
                                z_grad = torch.ones_like(z0)
                            gb_depth = torch.cat((z0, z_grad), dim=-1)
                            gb_depth = gb_depth.view(-1, gb_depth.shape[-1])

                    ks_flat = ks.view(h*w,3)
                    kd_flat = kd.view(h*w,3)

                    if self.opt.test and self.opt.envmap_path != 'None':
                        kd_flat[..., 0] = kd_flat[..., 0] * self.opt.albedo_scale_x
                        kd_flat[..., 1] = kd_flat[..., 1] * self.opt.albedo_scale_y
                        kd_flat[..., 2] = kd_flat[..., 2] * self.opt.albedo_scale_z
                        kd_flat = torch.clamp(kd_flat, min=0.0, max=1.0)

                    roughness_specular = torch.cat((ks_flat[...,1:2], ks_flat[...,2:3]), dim=-1)
                    gb_normal = gb_normal.view(-1, 3)
                    occ_map = mask.reshape(h*w,1)
                    big_ray_o = rays_o[0,:].expand(h*w,3)
                    depth_map = torch.norm(xyzs - big_ray_o, dim=1).view(h*w,1)
                    #'''restir buffers
                    prev_occ_map = torch.zeros((h * w, 1), dtype=torch.float, device='cuda')
                    prev_normal_depth = torch.zeros((h * w, 4), dtype=torch.float, device='cuda')
                    prev_brdf_map = torch.zeros((h * w, 3), dtype=torch.float, device='cuda')
                    prev_ray_dir = torch.zeros((h * w, 3), dtype=torch.float, device='cuda')
                    #'''
                    spp = self.opt.spp
                    de = 2
                    stepWidth = 2**(de-1)
                    c = 2.0
                    n = 0.1
                    p = 0.001
                    use_scale = False
                    if self.opt.test and self.opt.envmap_path != 'None':
                        use_scale = True

                    brdf_rgbs, brdf_diff, brdf_spec, brdf_indirect, brdf_indirect_diff, brdf_indirect_spec = run_restir_di_with_pt(
                                use_scale, self.opt.albedo_scale_x, self.opt.albedo_scale_y, self.opt.albedo_scale_z,
                                self.mlp_mat_opt, gb_depth,
                                self.bvh_restir_worker,
                                self.make_sampleable_m, self.generateLightTiles_m, self.InitialResampling_m, self.TemporalResampling_m, self.SpatialResampling_m, self.EvaluateFinalSamples_m, self.FinalShading_m,
                                self.denoising_m,
                                self.light_data, self.light_uv, self.light_inv_pdf, self.reservoirs, self.prev_reservoirs, self.final_samples, self.neighborOffsets,
                                self.light_tile_count, self.light_tile_size,
                                env_map, occ_map, gb_normal, depth_map, kd_flat, roughness_specular, dirs, xyzs if self.opt.enable_offset_nerf_grad else xyzs.detach(),
                                prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir,
                                w, h, spp, de, stepWidth, c, n, p)
                    
                    if self.opt.use_hdr:
                        brdf_rgbs = brdf_rgbs * (2**self.opt.exposure)
                        brdf_rgbs = torch.clamp(brdf_rgbs, min=0.0, max=1.0)
                    else:
                        brdf_rgbs = torch.clamp(brdf_rgbs, min=0.0, max=1.0)

                else:
                    ks_flat = ks.reshape(h*w,3)
                    kd_flat = kd.reshape(h*w,3)
                    roughness_map = ks_flat[:, 1:2].repeat(1,3)
                    fresnel_map = ks_flat[:, 2:3].repeat(1,3)
                    gb_normal = gb_normal.view(-1, 3)
                    #env_map.reshape(-1 , 3)
                    envmap_h = self.opt.light_probe_res_hw[0]
                    envmap_w = self.opt.light_probe_res_hw[1]

                    brdf_rgbs[mask_flatten], brdf_diff[mask_flatten], brdf_spec[mask_flatten] = dump_render(
                                                        self.intersector,
                                                        xyzs[mask_flatten] if self.opt.enable_offset_nerf_grad else xyzs[mask_flatten].detach(), 
                                                        gb_normal[mask_flatten], 
                                                        kd_flat[mask_flatten], roughness_map[mask_flatten], fresnel_map[mask_flatten], 
                                                        dirs[mask_flatten], env_map, envmap_h, envmap_w, 
                                                        self, sample_method='stratified_sampling', color_chunk_size=h*w, chunk_size=1600000, device='cuda',
                                                        use_linear2srgb=True
                                                        )

        if self.opt.use_brdf and not is_test and self.opt.lambda_extra_kd > 0:
            out_ao = torch.zeros((h*w, 3), dtype=torch.float, requires_grad=False, device='cuda')
            self.denoising_m.process_normal_ao(
                    framedim_x=int(w), framedim_y=int(h),
                    occ_map=occ_map, normal_map=gb_normal.detach(), ray_dir=dirs,
                    out_ao = out_ao
                    )\
                    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(w)+15)//16, (int(h)+15)//16, 1))
            
        rgbs = rgbs.view(1, h, w, 3)
        # handle brdf
        if self.opt.color_space == 'srgb':
            brdf_rgbs = torch.nan_to_num(brdf_rgbs, 0.0)
            brdf_rgbs[..., :3] = linear2srgb_torch(brdf_rgbs[..., :3])
            if self.opt.use_brdf:
                rgb_diff = srgb_to_linear(rgb_diff)
                rgb_spec = srgb_to_linear(rgb_spec)
        brdf_rgbs = brdf_rgbs.view(1, h, w, 3)
        if self.opt.use_brdf and not is_test:
            brdf_diff = brdf_diff.view(1, h, w, 3)
            brdf_spec = brdf_spec.view(1, h, w, 3)
            if self.opt.lambda_extra_kd > 0:
                out_ao = out_ao.view(1, h, w, 3)
            if self.opt.use_restir:
                brdf_indirect = brdf_indirect.view(1, h, w, 3)
                brdf_indirect_diff = brdf_indirect_diff.view(1, h, w, 3)
                brdf_indirect_spec = brdf_indirect_spec.view(1, h, w, 3)
            #for loss
            rgb_diff = rgb_diff.view(1, h, w, 3)
            rgb_spec = rgb_spec.view(1, h, w, 3)

        alphas = mask.float()
        
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        brdf_rgbs = dr.antialias(brdf_rgbs, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        if self.opt.use_brdf and not is_test:
            brdf_diff = dr.antialias(brdf_diff, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
            brdf_spec = dr.antialias(brdf_spec, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

            rgb_diff = dr.antialias(rgb_diff.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
            rgb_spec = dr.antialias(rgb_spec.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

            if self.opt.lambda_extra_kd > 0:
                out_ao = dr.antialias(out_ao.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

            if self.opt.use_restir:
                brdf_indirect = dr.antialias(brdf_indirect.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
                brdf_indirect_diff = dr.antialias(brdf_indirect_diff.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
                brdf_indirect_spec = dr.antialias(brdf_indirect_spec.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        
        if self.opt.use_brdf and is_test:
            rgb_diff = rgb_diff.view(1, h, w, 3)
            rgb_spec = rgb_spec.view(1, h, w, 3)
            rgb_diff = dr.antialias(rgb_diff.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
            rgb_spec = dr.antialias(rgb_spec.detach(), rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

        image = alphas * rgbs
        image_brdf = alphas * brdf_rgbs
        if self.opt.use_brdf and is_test:
            image_rgb_diff = alphas * rgb_diff
            image_rgb_spec = alphas * rgb_spec

            image_rgb_diff = image_rgb_diff.detach()
            image_rgb_spec = image_rgb_spec.detach()

        if self.opt.use_brdf and not is_test:
            image_brdf_diff = alphas * brdf_diff
            image_brdf_spec = alphas * brdf_spec

            image_rgb_diff = alphas * rgb_diff
            image_rgb_spec = alphas * rgb_spec

            image_rgb_diff = image_rgb_diff.detach()
            image_rgb_spec = image_rgb_spec.detach()

            if self.opt.lambda_extra_kd > 0:
                image_ao = alphas * out_ao
                image_ao = image_ao.detach()

            if self.opt.use_restir:
                image_brdf_indirect = alphas * brdf_indirect
                image_brdf_indirect = image_brdf_indirect.detach()

                image_brdf_indirect_diff = alphas * brdf_indirect_diff
                image_brdf_indirect_diff = image_brdf_indirect_diff.detach()
                image_brdf_indirect_spec = alphas * brdf_indirect_spec
                image_brdf_indirect_spec = image_brdf_indirect_spec.detach()

        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas

        # trig_id for updating trig errors
        trig_id = rast[0, :, :, -1] - 1 # [h, w]

        if self.opt.use_brdf and is_test:
            test_mask = ((mask.squeeze(dim=0))>0).repeat(1,1,3)
            ks_image = (ks.detach().squeeze(0))
            ks_image[~test_mask] = 0
            ks_image = ks_image + (1-alphas.detach())

            kd_image = (kd.detach().squeeze(0))
            kd_image[~test_mask] = 0
            kd_image = kd_image + (1-alphas.detach())

            normal_image = (gb_normal.detach().reshape(h,w,3))
            normal_image = normal_image + (1-alphas.detach())*torch.tensor([0.0, 0.0, 1.0], device=normal_image.device)
        if self.opt.use_brdf and not is_test:
            kd_grad = (kd_grad.squeeze(0))
            ks_grad = (ks_grad.squeeze(0))
            nrm_grad = (nrm_grad.squeeze(0))
            kd_image = (kd.squeeze(0))

        # ssaa
        if self.opt.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            image_brdf = scale_img_hwc(image_brdf, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest')

            if self.opt.use_brdf and is_test:
                ks_image = scale_img_hwc(ks_image, (h0, w0))
                kd_image = scale_img_hwc(kd_image, (h0, w0))
                normal_image = scale_img_hwc(normal_image, (h0, w0))
                image_rgb_diff = scale_img_hwc(image_rgb_diff, (h0, w0))
                image_rgb_spec = scale_img_hwc(image_rgb_spec, (h0, w0))

            if self.opt.use_brdf and not is_test:
                kd_grad = scale_img_hwc(kd_grad, (h0, w0))
                ks_grad = scale_img_hwc(ks_grad, (h0, w0))
                nrm_grad = scale_img_hwc(nrm_grad, (h0, w0))
                kd_image = scale_img_hwc(kd_image, (h0, w0))
                image_brdf_diff = scale_img_hwc(image_brdf_diff, (h0, w0))
                image_brdf_spec = scale_img_hwc(image_brdf_spec, (h0, w0))

                image_rgb_diff = scale_img_hwc(image_rgb_diff, (h0, w0))
                image_rgb_spec = scale_img_hwc(image_rgb_spec, (h0, w0))

                if self.opt.lambda_extra_kd > 0:
                    image_ao = scale_img_hwc(image_ao, (h0, w0))

                if self.opt.use_restir:
                    image_brdf_indirect = scale_img_hwc(image_brdf_indirect, (h0, w0))
                    image_brdf_indirect_diff = scale_img_hwc(image_brdf_indirect_diff, (h0, w0))
                    image_brdf_indirect_spec = scale_img_hwc(image_brdf_indirect_spec, (h0, w0))

        
        self.triangles_errors_id = trig_id

        image = image + T * bg_color
        image_brdf = image_brdf + T * bg_color
        
        image = image.view(*prefix, 3)
        image_brdf = image_brdf.view(*prefix, 3)
        depth = depth.view(*prefix)

        if self.opt.use_brdf and not is_test:
            #image_brdf_diff = image_brdf_diff + T * bg_color
            image_brdf_diff = image_brdf_diff.view(*prefix, 3)

            #image_brdf_spec = image_brdf_spec + T * bg_color
            image_brdf_spec = image_brdf_spec.view(*prefix, 3)

            image_rgb_diff = image_rgb_diff.view(*prefix, 3)
            image_rgb_spec = image_rgb_spec.view(*prefix, 3)

            if self.opt.lambda_extra_kd > 0:
                image_ao = image_ao.view(*prefix, 3)

            if self.opt.use_restir:
                image_brdf_indirect = image_brdf_indirect.view(*prefix, 3)
                image_brdf_indirect_diff = image_brdf_indirect_diff.view(*prefix, 3)
                image_brdf_indirect_spec = image_brdf_indirect_spec.view(*prefix, 3)

        results['depth'] = depth
        results['image'] = image
        results['image_brdf'] = image_brdf
        results['weights_sum'] = 1 - T

        if self.opt.use_brdf and is_test:

            image_rgb_diff = image_rgb_diff.view(*prefix, 3)
            image_rgb_spec = image_rgb_spec.view(*prefix, 3)

            image_rgb_diff = image_rgb_diff.view(h0, w0, 3)
            image_rgb_spec = image_rgb_spec.view(h0, w0, 3)

            results['kd'] = kd_image
            results['ks'] = ks_image#ks_image[...,1:2].repeat(1,1,3)
            results['normal'] = normal_image
            temp_env_map = env_map.detach()
            if self.opt.use_hdr:
                temp_env_map = temp_env_map * (2**self.opt.exposure)
            results['env_map'] = temp_env_map#generate_light_image(env_map.detach(), [128, 256])
            results['rgb_diffuse_light'] = image_rgb_diff
            results['rgb_specular_light'] = image_rgb_spec
        
        if self.opt.use_brdf and not is_test:
            results['kd_grad'] = kd_grad*(1-T)
            results['ks_grad'] = ks_grad*(1-T)
            results['normal_grad'] = nrm_grad*(1-T)
            results['kd'] = kd_image
            results['diffuse_light'] = image_brdf_diff
            results['specular_light'] = image_brdf_spec
            results['rgb_diffuse_light'] = image_rgb_diff
            results['rgb_specular_light'] = image_rgb_spec
            if self.opt.lambda_extra_kd > 0:
                results['normal_ao'] = image_ao
            if self.opt.use_restir:
                results['img_brdf_indirect'] = image_brdf_indirect
                results['img_brdf_indirect_diff'] = image_brdf_indirect_diff
                results['img_brdf_indirect_spec'] = image_brdf_indirect_spec

        torch.cuda.empty_cache()

        # tmp: visualize accumulated triangle error by abusing depth
        # error_val = self.triangles_errors[trig_id.view(-1)].view(*prefix)
        # error_cnt = self.triangles_errors_cnt[trig_id.view(-1)].view(*prefix)
        # cnt_mask = error_cnt > 0
        # error_val[cnt_mask] = error_val[cnt_mask] / error_cnt[cnt_mask].float()
        # results['depth'] = error_val

        return results

    @torch.no_grad()
    def update_triangles_errors(self, loss):
        # loss: [H, W], detached!

        # always call after render_stage1, so self.triangles_errors_id is not None.
        indices = self.triangles_errors_id.view(-1).long()
        mask = (indices >= 0)

        indices = indices[mask].contiguous()
        values = loss.view(-1)[mask].contiguous()

        global TORCH_SCATTER

        if TORCH_SCATTER is None:
            import torch_scatter
            TORCH_SCATTER = torch_scatter

        TORCH_SCATTER.scatter_add(values, indices, out=self.triangles_errors)
        TORCH_SCATTER.scatter_add(torch.ones_like(values), indices, out=self.triangles_errors_cnt)

        self.triangles_errors_id = None


    @torch.no_grad()
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.density_bitfield.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)
        
        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):

            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.to(device), 0, 1)).float().unsqueeze(0) # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(self.glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1
            
            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1 # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = (mask == 0) # unseen faces by all cameras

        print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
        
        return mask # [N]


    @torch.no_grad()
    def mark_untrained_grid(self, dataset, S=64):
        
        # data: reference to the dataset object

        poses = dataset.poses # [B, 4, 4]
        intrinsics = dataset.intrinsics # [4] or [B/1, 4]
        cam_near_far = dataset.cam_near_far if hasattr(dataset, 'cam_near_far') else None # [B, 2]
  
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if isinstance(intrinsics, np.ndarray):
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = torch.chunk(intrinsics, 4, dim=-1)
        
        mask_cam = torch.zeros_like(self.density_grid)
        mask_aabb = torch.zeros_like(self.density_grid)

        # pc = []
        poses = poses.to(mask_cam.device)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # first, mark out-of-AABB region
                        mask_min = (cas_world_xyzs >= (self.aabb_train[:3] - half_grid_size)).sum(-1) == 3
                        mask_max = (cas_world_xyzs <= (self.aabb_train[3:] + half_grid_size)).sum(-1) == 3
                        mask_aabb[cas, indices] += (mask_min & mask_max).reshape(-1)

                        # second, mark out-of-camera region
                        # split pose to batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            cam_xyzs[:, :, 2] *= -1 # crucial, camera forward is negative now...

                            if torch.is_tensor(fx):
                                cx_div_fx = cx[head:tail] / fx[head:tail]
                                cy_div_fy = cy[head:tail] / fy[head:tail]
                            else:
                                cx_div_fx = cx / fx
                                cy_div_fy = cy / fy
                            
                            min_near = self.opt.min_near if cam_near_far is None else cam_near_far[head:tail, 0].unsqueeze(1)
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > min_near # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx_div_fx * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy_div_fy * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask = (mask_z & mask_x & mask_y).sum(0).bool().reshape(-1) # [N]

                            # for visualization
                            # pc.append(cas_world_xyzs[0][mask])

                            # update mask_cam 
                            mask_cam[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[((mask_cam == 0) | (mask_aabb == 0))] = -1

        print(f'[mark untrained grid] {((mask_cam == 0) | (mask_aabb == 0)).sum()} from {self.grid_size ** 3 * self.cascade}')

    
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if self.opt.stage > 0:
            return

        if not self.cuda_ray:
            return 
        
        ### update density grid

        with torch.no_grad():

            tmp_grid = - torch.ones_like(self.density_grid)
        
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                                if self.opt.sdf:
                                    inv_s = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)
                                    sigmas = torch.sigmoid(- sigmas * inv_s) * inv_s
                            # assign 
                            tmp_grid[cas, indices] = sigmas

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            
        if not self.opt.trainable_density_grid:
            self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        else:
            # update grid via a loss term.
            loss = F.mse_loss(self.density_grid[valid_mask], tmp_grid[valid_mask])

            # cascaded reg
            loss_density = 0
            if self.opt.lambda_density > 0:
                for cas in range(1, self.cascade):
                    loss_density = loss_density + (2 ** (cas - 1)) * self.opt.lambda_density * self.density_grid[cas][valid_mask[cas]].mean()

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        # density_thresh = 0 if self.iter_density < 64 else self.density_thresh
        self.density_bitfield = raymarching.packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')

        if not self.opt.trainable_density_grid:
            return None
        else:
            return loss + loss_density
