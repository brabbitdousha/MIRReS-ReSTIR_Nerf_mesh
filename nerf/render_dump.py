import torch
import torch.nn.functional as F
import numpy as np

def safe_l2_normalize(x, dim=None, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

@torch.no_grad()
def batch_intersector(intersector, rays_o, rays_d, vis_near, chunk_size):
    device = rays_o.device
    visibility_compute = torch.ones((rays_o.shape[0]), dtype=torch.float32).to(device) # [N, 1]
    rays_o = rays_o + rays_d*vis_near # a little offset

    chunk_idxs_vis_compute = torch.split(torch.arange(rays_o.shape[0]), chunk_size)
    for chunk_idx in chunk_idxs_vis_compute:
        chunk_surface_pts = rays_o[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = rays_d[chunk_idx]    # [chunk_size, 3]
        hit, _, _, _, _, _ = intersector.intersects_closest(
            chunk_surface_pts, chunk_surf2light, stream_compaction=True
        )
        visibility_chunk = visibility_compute[chunk_idx]
        visibility_chunk[hit] = 0.0
        visibility_compute[chunk_idx] = visibility_chunk
    
    visibility_compute = visibility_compute.reshape(-1, 1)  # [N, 1]

    return visibility_compute

#----------------------------------------------------------------------------
# render_with_brdf from https://github.com/Haian-Jin/TensoIR/blob/main/models/relight_utils.py
#--------
def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec

# !!!
brdf_specular = GGX_specular

def get_light_rgbs(env_map, envmap_h, envmap_w, incident_light_directions=None, device='cuda'):
    remapped_light_directions = incident_light_directions.to(device).reshape(-1, 3) # [sample_number, 3]
    environment_map = env_map.reshape(envmap_h, envmap_w, 3).to(device) # [H, W, 3]
    environment_map = environment_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    phi = torch.arccos(remapped_light_directions[:, 2]).reshape(-1) - 1e-6
    theta = torch.atan2(remapped_light_directions[:, 1], remapped_light_directions[:, 0]).reshape(-1)
    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)

    light_rgbs = F.grid_sample(environment_map, grid, align_corners=False).squeeze().permute(1, 0).reshape(1, -1, 3)
    return light_rgbs

def dump_render(
        intersector,
        surface_xyz,
        normal_map,
        albedo_map,
        roughness_map,
        fresnel_map,
        rays_d,
        env_map,
        env_h,
        env_w,
        model,
        sample_method='stratified_sampling',
        color_chunk_size=15000,
        chunk_size=15000,
        device='cuda',
        use_linear2srgb=True
):
    diff_color = torch.zeros((surface_xyz.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]
    spec_color = torch.zeros((surface_xyz.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]
    brdf_color = torch.zeros((surface_xyz.shape[0], 3), dtype=torch.float32).to(device) # [N, 1]

    chunk_idxs_compute = torch.split(torch.arange(surface_xyz.shape[0]), color_chunk_size)
    for chunk_idx in chunk_idxs_compute:
        chunk_surface_xyz = surface_xyz[chunk_idx]  # [chunk_size, 3]
        chunk_normal_map = normal_map[chunk_idx]    # [chunk_size, 3]
        chunk_albedo_map = albedo_map[chunk_idx]    # [chunk_size, 3]
        chunk_roughness_map = roughness_map[chunk_idx]    # [chunk_size, 3]
        chunk_fresnel_map = fresnel_map[chunk_idx]    # [chunk_size, 3]
        chunk_rays_d = rays_d[chunk_idx]    # [chunk_size, 3]

        light_chunk, diff_light_chunk, spec_light_chunk = dump_render_run_mesh(intersector,
                                                chunk_surface_xyz, 
                                                chunk_normal_map, 
                                                chunk_albedo_map, chunk_roughness_map, chunk_fresnel_map, 
                                                chunk_rays_d, env_map, env_h, env_w, 
                                                model, sample_method, chunk_size, device,
                                                use_linear2srgb
                                                )

        brdf_color[chunk_idx] = light_chunk
        diff_color[chunk_idx] = diff_light_chunk
        spec_color[chunk_idx] = spec_light_chunk

    diff_color = diff_color.reshape(-1, 3)  # [N, 3]
    spec_color = spec_color.reshape(-1, 3)  # [N, 3]

    brdf_color = brdf_color.reshape(-1, 3)  # [N, 3]
    brdf_color = torch.clamp(brdf_color, min=0.0, max=1.0)  

    return brdf_color, diff_color, spec_color
    
def dump_render_run_mesh(
        intersector,
        surface_xyz,
        normal_map,
        albedo_map,
        roughness_map,
        fresnel_map,
        rays_d,
        env_map,
        env_h,
        env_w,
        model,
        sample_method='stratified_sampling',
        chunk_size=15000,
        device='cuda',
        use_linear2srgb=True
):
    # Relight module
    ## Sample surface points using depth prediction
    rays_d = rays_d.to(device)  # [bs, 3]
    light_idx = torch.zeros((rays_d.shape[0], 1), dtype=torch.int, device=device)
    ## Get incident light direction
    light_area_weight = model.light_area_weight.to(device) # [envW * envH, ]
    incident_light_dirs = model.fixed_viewdirs.to(device)  # [envW * envH, 3]
    surf2l = incident_light_dirs.reshape(1, -1, 3).repeat(surface_xyz.shape[0], 1, 1)  # [bs, envW * envH, 3]
    surf2c = -rays_d  # [bs, 3]
    surf2c =  F.normalize(surf2c, p=2, dim=-1, eps=1e-6)  # [bs, 3]

    ## get visibilty map from visibility network or compute it using density
    cosine = torch.einsum("ijk,ik->ij", surf2l, normal_map)  # surf2l:[bs, envW * envH, 3] * normal_map:[bs, 3] -> cosine:[bs, envW * envH]
    cosine = torch.clamp(cosine, min=0.0)  # [bs, envW * envH]
    cosine_mask = (cosine > 1e-6)  # [bs, envW * envH], mask half of the incident light that is behind the surface
    visibility_compute = torch.ones((*cosine_mask.shape, 1), device=device)   # [bs, envW * envH, 1]
    #indirect_light = torch.zeros((*cosine_mask.shape, 3), device=device)   # [bs, envW * envH, 3]

    vis_near = 0.001
    vis_rays_o = surface_xyz.unsqueeze(1).expand(-1, surf2l.shape[1], -1)[cosine_mask]
    vis_rays_d = surf2l[cosine_mask]
    visibility_compute[cosine_mask] = batch_intersector(intersector, vis_rays_o, vis_rays_d, vis_near, chunk_size)
    visibility_to_use = visibility_compute
    ## Get BRDF specs
    nlights = surf2l.shape[1]
    specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)  # [bs, envW * envH, 3]
    surface_brdf_diff_ = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi # [bs, envW * envH, 3]
    surface_brdf_diff = 1/np.pi
    surface_brdf_spec = specular
    surface_brdf = surface_brdf_diff_ + surface_brdf_spec

    ## Compute rendering equation
    envir_map_light_rgbs = get_light_rgbs(env_map, env_h, env_w, incident_light_dirs, device=device).to(device) # [light_num, envW * envH, 3]
    direct_light_rgbs = torch.index_select(envir_map_light_rgbs, dim=0, index=light_idx.squeeze(-1)).to(device) # [bs, envW * envH, 3]
    
    light_rgbs = visibility_to_use * direct_light_rgbs #+ indirect_light # [bs, envW * envH, 3]

    # # no visibility and indirect light
    # light_rgbs = direct_light_rgbs

    # # # no indirect light
    # light_rgbs = visibility_to_use * direct_light_rgbs  # [bs, envW * envH, 3]

    if sample_method == 'stratifed_sample_equal_areas':
        rgb_with_brdf_diff = torch.mean(4 * torch.pi * surface_brdf_diff * light_rgbs * cosine[:, :, None], dim=1)  # [bs, 3]
        rgb_with_brdf_spec = torch.mean(4 * torch.pi * surface_brdf_spec * light_rgbs * cosine[:, :, None], dim=1)  # [bs, 3]
        rgb_with_brdf = torch.mean(4 * torch.pi * surface_brdf * light_rgbs * cosine[:, :, None], dim=1)  # [bs, 3]
    else:
        light_pix_contrib_diff = surface_brdf_diff * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
        rgb_with_brdf_diff = torch.sum(light_pix_contrib_diff, dim=1)  # [bs, 3]
        light_pix_contrib_spec = surface_brdf_spec * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
        rgb_with_brdf_spec = torch.sum(light_pix_contrib_spec, dim=1)  # [bs, 3]
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
        rgb_with_brdf = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]

    ### Tonemapping
    #rgb_with_brdf = rgb_with_brdf_diff + rgb_with_brdf_spec
    #rgb_with_brdf = torch.clamp(rgb_with_brdf, min=0.0, max=1.0)  
    ### Colorspace transform
    #if use_linear2srgb and rgb_with_brdf.shape[0] > 0:
    #    rgb_with_brdf = linear2srgb_torch(rgb_with_brdf)
    
    return rgb_with_brdf, rgb_with_brdf_diff, rgb_with_brdf_spec

def dump_render_run(
        surface_xyz,
        normal_map,
        albedo_map,
        roughness_map,
        fresnel_map,
        rays_d,
        env_map,
        env_h,
        env_w,
        model,
        sample_method='stratified_sampling',
        chunk_size=15000,
        device='cuda',
        use_linear2srgb=True
):
    # Relight module
    ## Sample surface points using depth prediction
    rays_d = rays_d.to(device)  # [bs, 3]
    light_idx = torch.zeros((rays_d.shape[0], 1), dtype=torch.int, device=device)
    ## Get incident light direction
    light_area_weight = model.light_area_weight.to(device) # [envW * envH, ]
    incident_light_dirs = model.fixed_viewdirs.to(device)  # [envW * envH, 3]
    surf2l = incident_light_dirs.reshape(1, -1, 3).repeat(surface_xyz.shape[0], 1, 1)  # [bs, envW * envH, 3]
    surf2c = -rays_d  # [bs, 3]
    surf2c =  F.normalize(surf2c, p=2, dim=-1, eps=1e-6)  # [bs, 3]

    ## get visibilty map from visibility network or compute it using density
    cosine = torch.einsum("ijk,ik->ij", surf2l, normal_map)  # surf2l:[bs, envW * envH, 3] * normal_map:[bs, 3] -> cosine:[bs, envW * envH]
    cosine = torch.clamp(cosine, min=0.0)  # [bs, envW * envH]
    cosine_mask = (cosine > 1e-6)  # [bs, envW * envH], mask half of the incident light that is behind the surface
    visibility_compute = torch.zeros((*cosine_mask.shape, 1), device=device)   # [bs, envW * envH, 1]
    indirect_light = torch.zeros((*cosine_mask.shape, 3), device=device)   # [bs, envW * envH, 3]

    vis_near = 0.02
    vis_far = 1.5
    chunk_size = 160000
    visibility_compute[cosine_mask], \
        indirect_light[cosine_mask] = model.compute_secondary_shading_effects(
                                            surface_pts=surface_xyz.unsqueeze(1).expand(-1, surf2l.shape[1], -1)[cosine_mask],
                                            surf2light=surf2l[cosine_mask],
                                            chunk_size=chunk_size, 
                                            vis_near=vis_near, 
                                            vis_far=vis_far, 
                                            index=None, dt_gamma=0, bg_color=None, perturb=False, max_steps=1024, T_thresh=1e-4, cam_near_far=None, shading='full'
                                            )
    visibility_to_use = visibility_compute
    ## Get BRDF specs
    nlights = surf2l.shape[1]
    specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)  # [bs, envW * envH, 3]
    surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular # [bs, envW * envH, 3]


    ## Compute rendering equation
    envir_map_light_rgbs = get_light_rgbs(env_map, env_h, env_w, incident_light_dirs, device=device).to(device) # [light_num, envW * envH, 3]
    direct_light_rgbs = torch.index_select(envir_map_light_rgbs, dim=0, index=light_idx.squeeze(-1)).to(device) # [bs, envW * envH, 3]
    
    light_rgbs = visibility_to_use * direct_light_rgbs #+ indirect_light # [bs, envW * envH, 3]

    # # no visibility and indirect light
    # light_rgbs = direct_light_rgbs

    # # # no indirect light
    # light_rgbs = visibility_to_use * direct_light_rgbs  # [bs, envW * envH, 3]

    if sample_method == 'stratifed_sample_equal_areas':
        rgb_with_brdf = torch.mean(4 * torch.pi * surface_brdf * light_rgbs * cosine[:, :, None], dim=1)  # [bs, 3]

    else:
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
        rgb_with_brdf = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
    ### Tonemapping
    rgb_with_brdf = torch.clamp(rgb_with_brdf, min=0.0, max=1.0)  
    ### Colorspace transform
    #if use_linear2srgb and rgb_with_brdf.shape[0] > 0:
    #    rgb_with_brdf = linear2srgb_torch(rgb_with_brdf)
    

    return rgb_with_brdf