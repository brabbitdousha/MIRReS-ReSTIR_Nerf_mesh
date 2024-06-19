import numpy as np
import time
import torch
import torchvision.utils as vutils
import slangpy
import pyexr
from nerf.ScreenSpaceReSTIR.Denoising import *
from nerf.ScreenSpaceReSTIR.GenerateLightTiles import *
from nerf.ScreenSpaceReSTIR.Resampling import *
from .render_dump import batch_intersector, safe_l2_normalize
from .renderutils.ops import bilateral_denoiser, bilateral_denoiser_no_di

class restirbvhWorker:
    def __init__(self, vt, vt_ind):
        self.vrt = vt
        self.v_ind = vt_ind

        print("loading bvh shaders, this can be slow at first time...")
        self.m_gen_ele = slangpy.loadModule('nerf/bvhworkers/get_elements.slang')
        self.m_morton_codes = slangpy.loadModule('nerf/bvhworkers/lbvh_morton_codes.slang')
        self.m_radixsort = slangpy.loadModule('nerf/bvhworkers/lbvh_single_radixsort.slang')
        self.m_hierarchy = slangpy.loadModule('nerf/bvhworkers/lbvh_hierarchy.slang')
        self.m_bounding_box = slangpy.loadModule('nerf/bvhworkers/lbvh_bounding_boxes.slang')

    def update_bvh(self):
        #first part, get element and bbox---------------
        primitive_num = self.v_ind.shape[0]
        ele_primitiveIdx = torch.zeros((primitive_num, 1), dtype=torch.int, device='cuda')
        ele_aabb = torch.zeros((primitive_num, 6), dtype=torch.float, device='cuda')

        # Invoke normally
        self.m_gen_ele.generateElements(vert=self.vrt, v_indx=self.v_ind, ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb)\
            .launchRaw(blockSize=(256, 1, 1), gridSize=((primitive_num+255)//256, 1, 1))
        extent_min_x = ele_aabb[:,0].min()
        extent_min_y = ele_aabb[:,1].min()
        extent_min_z = ele_aabb[:,2].min()

        extent_max_x = ele_aabb[:,3].max()
        extent_max_y = ele_aabb[:,4].max()
        extent_max_z = ele_aabb[:,5].max()
        num_ELEMENTS = ele_aabb.shape[0]
        #-------------------------------------------------
        #morton codes part
        pcMortonCodes = self.m_morton_codes.pushConstantsMortonCodes(
            g_num_elements=num_ELEMENTS, g_min_x=extent_min_x, g_min_y=extent_min_y, g_min_z=extent_min_z,
            g_max_x=extent_max_x, g_max_y=extent_max_y, g_max_z=extent_max_z
        )
        morton_codes_ele = torch.zeros((num_ELEMENTS, 2), dtype=torch.int, device='cuda')

        self.m_morton_codes.morton_codes(pc=pcMortonCodes, ele_aabb=ele_aabb, morton_codes_ele=morton_codes_ele)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

        #--------------------------------------------------
        # radix sort part
        morton_codes_ele_pingpong = torch.zeros((num_ELEMENTS, 2), dtype=torch.int, device='cuda')
        self.m_radixsort.radix_sort(g_num_elements=int(num_ELEMENTS), g_elements_in=morton_codes_ele, g_elements_out=morton_codes_ele_pingpong)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))

        #--------------------------------------------------
        # hierarchy
        num_LBVH_ELEMENTS = num_ELEMENTS + num_ELEMENTS - 1
        LBVHNode_info = torch.zeros((num_LBVH_ELEMENTS, 3), dtype=torch.int, device='cuda')
        LBVHNode_aabb = torch.zeros((num_LBVH_ELEMENTS, 6), dtype=torch.float, device='cuda')
        LBVHConstructionInfo = torch.zeros((num_LBVH_ELEMENTS, 2), dtype=torch.int, device='cuda')

        self.m_hierarchy.hierarchy(g_num_elements=int(num_ELEMENTS), ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb,
                            g_sorted_morton_codes=morton_codes_ele, g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, g_lbvh_construction_infos=LBVHConstructionInfo)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

        #--------------------------------------------------
        # bounding_boxes
        #'''
        tree_heights = torch.zeros((num_ELEMENTS, 1), dtype=torch.int, device='cuda')
        self.m_bounding_box.get_bvh_height(g_num_elements=int(num_ELEMENTS), g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                    g_lbvh_construction_infos=LBVHConstructionInfo, tree_heights=tree_heights)\
        .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

        tree_height_max = tree_heights.max()
        for i in range(tree_height_max):
            self.m_bounding_box.get_bbox(g_num_elements=int(num_ELEMENTS), expected_height=int(i+1),
                                g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                        g_lbvh_construction_infos=LBVHConstructionInfo)\
            .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

        self.m_bounding_box.set_root(
                    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb)\
            .launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1)) 
        
        return LBVHNode_info, LBVHNode_aabb

    def update_mesh(self, vt, vt_ind):
        self.vrt = vt
        self.v_ind = vt_ind
        self.LBVHNode_info, self.LBVHNode_aabb = self.update_bvh()
    
    def InitialResampling_(self, m, pos_map,
                       reservoirs,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir, pdf_, cdf_, mpdf_, mcdf_,
                       light_data, light_uv, light_inv_pdf
                       ):

        m.process_InitialResampling_(
        g_lbvh_info=self.LBVHNode_info, g_lbvh_aabb=self.LBVHNode_aabb, vert=self.vrt, v_indx=self.v_ind, pos_map=pos_map,
        reservoirs=reservoirs, env_tex=env_tex,
        env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y, frameIndex=frameIndex,
        occ_map=occ_map, normal_depth=normal_depth, brdf_map=brdf_map, ray_dir=ray_dir,
        pdf_=pdf_, cdf_=cdf_, mpdf_=mpdf_, mcdf_=mcdf_,
        light_data=light_data, light_uv=light_uv, light_inv_pdf=light_inv_pdf
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return 'hello'
    
    def SpatialResampling_(self, m, pos_map,
                       reservoirs, prev_reservoirs, neighborOffsets,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir
                       ):

        m.process_SpatialResampling_(
        g_lbvh_info=self.LBVHNode_info, g_lbvh_aabb=self.LBVHNode_aabb, vert=self.vrt, v_indx=self.v_ind, pos_map=pos_map,
        reservoirs=reservoirs, prevReservoirs=prev_reservoirs, neighborOffsets=neighborOffsets,
        env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y, frameIndex=frameIndex,
        occ_map=occ_map, normal_depth=normal_depth, brdf_map=brdf_map, ray_dir=ray_dir
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return 'hello'
    
    def EvaluateFinalSamples_get_vis(self, m,
                       pos_map,
                       reservoirs,
                       framedim_x, framedim_y,
                       vis_map
                       ):
        m.process_EvaluateFinalSamples_get_vis(g_lbvh_info=self.LBVHNode_info, g_lbvh_aabb=self.LBVHNode_aabb, vert=self.vrt, v_indx=self.v_ind,
        reservoirs=reservoirs,
        framedim_x=framedim_x, framedim_y=framedim_y,
        pos_map=pos_map, vis_map=vis_map
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return 'hello'

def load_m_for_restir(framedim_x, framedim_y):
#load part
    make_sampleable_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/make_sampleable.slang')
    light_tile_count = 128
    light_tile_size = 1024
    generateLightTiles_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/GenerateLightTiles.slang',
                                              defines={"LIGHT_TILE_COUNT": int(light_tile_count),
                                                       "LIGHT_TILE_SIZE": int(light_tile_size)
                                                       }
                                              )
    screen_tile_size = 8
    initialLightSampleCount = 32
    initialBRDFSampleCount = 1
    InitialResampling_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/InitialResampling.slang',
                                              defines={"LIGHT_TILE_COUNT": int(light_tile_count),
                                                       "LIGHT_TILE_SIZE": int(light_tile_size),
                                                       "SCREEN_TILE_SIZE": int(screen_tile_size),
                                                       "INITIAL_LIGHT_SAMPLE_COUNT": int(initialLightSampleCount),
                                                       "INITIAL_BRDF_SAMPLE_COUNT": int(initialBRDFSampleCount)
                                                       }
                                              )
    maxHistoryLength = 20
    TemporalResampling_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/TemporalResampling.slang',
                                              defines={"MAX_HISTORY_LENGTH": int(maxHistoryLength)
                                                       }
                                              )
    spatial_NEIGHBOR_OFFSET_COUNT = 8192
    spatialNeighborCount = 5
    spatialGatherRadius = 30
    SpatialResampling_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/SpatialResampling.slang',
                                              defines={"NEIGHBOR_OFFSET_COUNT": int(spatial_NEIGHBOR_OFFSET_COUNT),
                                                       "NEIGHBOR_COUNT": int(spatialNeighborCount),
                                                       "GATHER_RADIUS": int(spatialGatherRadius)
                                                       }
                                              )
    EvaluateFinalSamples_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/EvaluateFinalSamples.slang')

    FinalShading_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/FinalShading.slang')

    denoising_m = slangpy.loadModule('nerf/ScreenSpaceReSTIR/EAWDenoise.slang')

    light_data = torch.zeros((light_tile_count*light_tile_size, 3), dtype=torch.float, device='cuda')
    light_uv = torch.zeros((light_tile_count*light_tile_size, 2), dtype=torch.int, device='cuda')
    light_inv_pdf = torch.zeros((light_tile_count*light_tile_size, 1), dtype=torch.float, device='cuda')

    res_light_data = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    res_light_pdf = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    res_M = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.int, device='cuda')
    res_weight = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')

    reservoirs = (res_light_data, res_light_pdf,
                             res_M, res_weight)
    
    final_dir = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    final_dis = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    final_Li = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')

    final_samples = (final_dir, final_dis, final_Li)
    
    prev_res_light_data = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    prev_res_light_pdf = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    prev_res_M = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.int, device='cuda')
    prev_res_weight = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')

    prev_reservoirs = (prev_res_light_data, prev_res_light_pdf,
                             prev_res_M, prev_res_weight)
    
    # Create neighbor offset texture.
    start_time = time.time()
    neighborOffsets = torch.zeros((spatial_NEIGHBOR_OFFSET_COUNT*2, 1), dtype=torch.float, device='cuda')
    make_sampleable_m.createNeighborOffsetTexture(sampleCount=spatial_NEIGHBOR_OFFSET_COUNT, neighborOffsets=neighborOffsets)\
    .launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))
    neighborOffsets = neighborOffsets.reshape(-1,2)
    neighborOffsets = neighborOffsets/127
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Create neighbor offset time consumed: {elapsed_time} s")
    return (make_sampleable_m, generateLightTiles_m, InitialResampling_m, TemporalResampling_m, SpatialResampling_m, EvaluateFinalSamples_m, FinalShading_m,
            denoising_m,
    light_data, light_uv, light_inv_pdf, reservoirs, prev_reservoirs, final_samples, neighborOffsets,
    light_tile_count, light_tile_size)

def restir_di_with_pt(
        use_scale,
        scale_x, scale_y, scale_z,
        mlp_mat,
        bvh_restir_worker,
        spp, framedim_x, framedim_y, 
        make_sampleable_m, generateLightTiles_m, InitialResampling_m, TemporalResampling_m, SpatialResampling_m, EvaluateFinalSamples_m, FinalShading_m,
        light_data, light_uv, light_inv_pdf, reservoirs, prev_reservoirs, final_samples, neighborOffsets, light_tile_count, light_tile_size,
        env_map_init, occ_map, pos_map, normal_map, depth_map, diffuse_map, roughness_specular, ray_dir_map,
        prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir, motionVectors, color):
    
    #common setting
    mTotalRISPasses = 5 + 15 #+20 for 3
    mFrameIndex = 0
    mCurRISPass = 0
    random_offset = np.random.randint(2**20)
    # visibility setting
    vis_near = 0.01

    #------
    
    #start!!
    total_color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    total_diff_light = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    total_spec_light = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    total_indirect_light = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')

    #indirect buffer
    total_color_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    color_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    color_diff_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    color_spec_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    total_diff_light_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    total_spec_light_1 = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')

    prd = torch.zeros((framedim_x*framedim_y, 5), dtype=torch.float, device='cuda')
    new_pos_map = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    new_ray_d = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    new_occ_map = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    new_normal_map = torch.zeros_like(normal_map)
    new_diffuse_map = torch.zeros_like(diffuse_map)
    new_roughness_specular = torch.zeros_like(roughness_specular)

    new_pos_map_temp = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    new_ray_d_temp = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    new_occ_map_temp = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    new_normal_map_temp = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    #---
    
    normal_depth = torch.cat((normal_map, depth_map), dim = -1)
    normal_depth = normal_depth.detach()
    brdf_map = torch.cat((diffuse_map[:,0:1] * 0.2126 + diffuse_map[:,1:2] * 0.7152 + diffuse_map[:,2:3] * 0.0722, 
                         roughness_specular[:,1:2] * 0.2126 + roughness_specular[:,1:2] * 0.7152 + roughness_specular[:,1:2] * 0.0722,
                         roughness_specular[:,0:1]), dim = -1)

    brdf_map = brdf_map.detach()
    brdf_map[:,2].clamp_(min=0.01, max=1)
    brdf_map[:,2] = brdf_map[:,2]*brdf_map[:,2]

    eva_vis_map = torch.ones((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    
    prev_res_light_data = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    prev_res_light_pdf = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')
    prev_res_M = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.int, device='cuda')
    prev_res_weight = torch.zeros((framedim_x*framedim_y, 1), dtype=torch.float, device='cuda')

    prev_reservoirs = (prev_res_light_data, prev_res_light_pdf,
                             prev_res_M, prev_res_weight)
    
    prev_occ_map = torch.zeros(occ_map.shape, dtype=torch.float, device='cuda')
    prev_normal_depth = torch.zeros(int(normal_map.shape[0]), (normal_map.shape[-1]+depth_map.shape[-1]), dtype=torch.float, device='cuda')
    prev_brdf_map = torch.zeros(brdf_map.shape, dtype=torch.float, device='cuda')
    prev_ray_dir = torch.zeros(ray_dir_map.shape, dtype=torch.float, device='cuda')

    #env map part
    env_map = env_map_init.detach()
    env_map_init = torch.flip(env_map_init, dims=[0]).reshape(-1,env_map_init.shape[2])
    width = env_map.shape[1]
    height = env_map.shape[0]
    debug_out = torch.zeros((1), dtype=torch.float, device='cuda')

    env_map = torch.flip(env_map, dims=[0]).reshape(-1,env_map.shape[2])
    pdf_, cdf_, mpdf_, mcdf_ = make_sampleable(make_sampleable_m, env_map, width, height)
    bounce_count = 0
    for i in range(0, spp):
        mCurRISPass = 0
        #light tiles part--------
        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        #'''
        GenerateLightTiles(generateLightTiles_m,
                        debug_out,
                        env_map, pdf_, cdf_, mpdf_, mcdf_,
                        width, height,
                        frameIndex,
                        light_data,
                        light_uv,
                        light_inv_pdf,
                        light_tile_count = light_tile_count,
                        light_tile_size = light_tile_size)
        mCurRISPass += 2

        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        bvh_restir_worker.InitialResampling_(InitialResampling_m, pos_map,
                        reservoirs,
                       env_map, width, height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir_map, pdf_, cdf_, mpdf_, mcdf_,
                       light_data, light_uv, light_inv_pdf
                       )

        mCurRISPass += 1
        #'''
        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        if (i > 0):
            TemporalResampling(TemporalResampling_m,
                            reservoirs, prev_reservoirs,
                            env_map, width, height,
                            framedim_x, framedim_y, frameIndex,
                            occ_map, normal_depth, brdf_map, ray_dir_map,
                            prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir,
                            motionVectors
                            )
            mCurRISPass += 1
        #'''
        #'''
        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        #swap
        #debug_out = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float).cuda()
        reservoirs, prev_reservoirs = prev_reservoirs, reservoirs
    
        bvh_restir_worker.SpatialResampling_(SpatialResampling_m, pos_map,
                       reservoirs, prev_reservoirs, neighborOffsets,
                       env_map, width, height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir_map
                       )
        mCurRISPass += 1

        bvh_restir_worker.EvaluateFinalSamples_get_vis(EvaluateFinalSamples_m, pos_map,
                                     reservoirs, framedim_x, framedim_y, eva_vis_map)
        #'''
        final_Li = EvaluateFinalSamples_di.apply( EvaluateFinalSamples_m,
                        reservoirs[0], reservoirs[1], reservoirs[2], reservoirs[3],
                        env_map_init, width, height,
                        framedim_x, framedim_y,
                        final_samples[0], final_samples[1], eva_vis_map
                        )

        color, color_diff, color_spec = FinalShading.apply(FinalShading_m,
                    final_samples[0],final_samples[1], final_Li,
                    env_map, width, height,
                    framedim_x, framedim_y,
                    occ_map, normal_map, ray_dir_map,
                    diffuse_map, roughness_specular
                       )

        #'''deal with indirect light
        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        process_new_dir_for_pt(
                    FinalShading_m,
                    bvh_restir_worker.LBVHNode_info, bvh_restir_worker.LBVHNode_aabb, bvh_restir_worker.vrt, bvh_restir_worker.v_ind,
                    frameIndex, bounce_count, framedim_x, framedim_y, 
                     occ_map, pos_map, normal_map.detach(), ray_dir_map, prd,
                     diffuse_map.detach(), roughness_specular.detach(),
                     new_pos_map, new_ray_d, new_occ_map, new_normal_map)
        mCurRISPass+=5
        #first bounce
        #'''
        indices = torch.where(new_occ_map>=0.5)
        kd_ks = mlp_mat.sample_no_di(new_pos_map[indices[0]])

        new_diffuse_map[indices[0]] = kd_ks[..., 0:3]
        new_roughness_specular[indices[0]] = torch.cat((kd_ks[...,4:5], kd_ks[...,5:6]), dim=-1)

        if use_scale:
            new_diffuse_map[indices[0],0] = new_diffuse_map[indices[0],0]*scale_x
            new_diffuse_map[indices[0],1] = new_diffuse_map[indices[0],1]*scale_y
            new_diffuse_map[indices[0],2] = new_diffuse_map[indices[0],2]*scale_z
            new_diffuse_map = torch.clamp(new_diffuse_map, min=0.0, max=1.0)

        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass
        indirect_one_hit_divided_no_grad(
                    FinalShading_m,
                    bvh_restir_worker.LBVHNode_info, bvh_restir_worker.LBVHNode_aabb, bvh_restir_worker.vrt, bvh_restir_worker.v_ind,
                    frameIndex, bounce_count+1, framedim_x, framedim_y, 
                     env_map, width, height, pdf_, cdf_, mpdf_, mcdf_,
                     new_occ_map, new_pos_map, new_normal_map, new_ray_d, prd,
                     new_diffuse_map, new_roughness_specular, color_1, color_diff_1, color_spec_1,
                     new_pos_map_temp, new_ray_d_temp, new_occ_map_temp, new_normal_map_temp)
        
        total_color_1+= color_1
        total_diff_light_1 += color_diff_1
        total_spec_light_1 += color_spec_1
        mCurRISPass+=5
        #'''

        #second bounce
        #'''
        indices = torch.where(new_occ_map_temp>=0.5)
        kd_ks = mlp_mat.sample_no_di(new_pos_map_temp[indices[0]])

        new_diffuse_map[indices[0]] = kd_ks[..., 0:3]
        new_roughness_specular[indices[0]] = torch.cat((kd_ks[...,4:5], kd_ks[...,5:6]), dim=-1)

        if use_scale:
            new_diffuse_map[indices[0],0] = new_diffuse_map[indices[0],0]*scale_x
            new_diffuse_map[indices[0],1] = new_diffuse_map[indices[0],1]*scale_y
            new_diffuse_map[indices[0],2] = new_diffuse_map[indices[0],2]*scale_z
            new_diffuse_map = torch.clamp(new_diffuse_map, min=0.0, max=1.0)

        frameIndex = random_offset + mTotalRISPasses * mFrameIndex + mCurRISPass

        indirect_one_hit_divided_no_grad(
                    FinalShading_m,
                    bvh_restir_worker.LBVHNode_info, bvh_restir_worker.LBVHNode_aabb, bvh_restir_worker.vrt, bvh_restir_worker.v_ind,
                    frameIndex, bounce_count+2, framedim_x, framedim_y, 
                     env_map, width, height, pdf_, cdf_, mpdf_, mcdf_,
                     new_occ_map_temp, new_pos_map_temp, new_normal_map_temp, new_ray_d_temp, prd,
                     new_diffuse_map, new_roughness_specular, color_1, color_diff_1, color_spec_1,
                     new_pos_map, new_ray_d, new_occ_map, new_normal_map)
        total_color_1+= color_1
        total_diff_light_1 += color_diff_1
        total_spec_light_1 += color_spec_1
        mCurRISPass+=5
        #'''


        #'''
        #swap
        mFrameIndex = mFrameIndex + 1
        reservoirs, prev_reservoirs = prev_reservoirs, reservoirs

        prev_occ_map = occ_map
        prev_normal_depth = normal_depth
        prev_brdf_map = brdf_map
        prev_ray_dir = ray_dir_map
        total_color+= color
        total_diff_light+= color_diff
        total_spec_light+= color_spec

        
    return total_color, total_color_1, total_diff_light, total_spec_light, total_diff_light_1, total_spec_light_1, total_indirect_light, mFrameIndex

def run_restir_di_with_pt(
        use_scale,
        scale_x, scale_y, scale_z,
        mlp_mat, gb_depth, bvh_restir_worker,
        make_sampleable_m, generateLightTiles_m, InitialResampling_m, TemporalResampling_m, SpatialResampling_m, EvaluateFinalSamples_m, FinalShading_m,
        denoising_m,
        light_data, light_uv, light_inv_pdf, reservoirs, prev_reservoirs, final_samples, neighborOffsets,
        light_tile_count, light_tile_size,
        env_map, occ_map, normal_map, depth_map, diffuse_map, roughness_specular, ray_dir_map, pos_map,
        prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir,
        framedim_x, framedim_y, spp, denoise_iter, stepWidth, c_phi_scale=1.0, n_phi_scale=0.1, p_phi_scale=0.1):
    indices = torch.where(occ_map<=0.5)
    occ_map[indices[0],:] = 0
    ray_dir_map = safe_l2_normalize(ray_dir_map, dim=-1)
    motionVectors = torch.zeros((framedim_x*framedim_y, 2), dtype=torch.float, device='cuda')#TODO: need to be changed
    color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    
    diffuse = diffuse_map
    normal = normal_map
    roughnessSpecular = roughness_specular
    
    total_color, total_color_1, total_diff_light, total_spec_light, total_diff_light_1, total_spec_light_1, total_indirect_light, mFrameIndex = restir_di_with_pt(
                                                                                                    use_scale,
                                                                                                    scale_x, scale_y, scale_z,
                                                                                                    mlp_mat,
                                                                                                    bvh_restir_worker,
                                                                                                   spp, framedim_x, framedim_y, 
                make_sampleable_m, generateLightTiles_m, InitialResampling_m, TemporalResampling_m, SpatialResampling_m, EvaluateFinalSamples_m, FinalShading_m,
                light_data, light_uv, light_inv_pdf, reservoirs, prev_reservoirs, final_samples, neighborOffsets, light_tile_count, light_tile_size,
                env_map, occ_map, pos_map, normal, depth_map, diffuse, roughnessSpecular, ray_dir_map,
                prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir, motionVectors, color)

        #get_gpu_mem('after_restir_di')
        #color = color.reshape(framedim_y, framedim_x, 3)
    total_color = (total_color)/mFrameIndex
    total_diff_light = (total_diff_light)/mFrameIndex
    total_spec_light = (total_spec_light)/mFrameIndex

    total_color_1 = (total_color_1)/mFrameIndex #indirect color
    total_diff_light_1 = (total_diff_light_1)/mFrameIndex
    total_spec_light_1 = (total_spec_light_1)/mFrameIndex

    combined_color_indirect = total_diff_light_1 + total_spec_light_1

    if gb_depth is None:
        denoised_diffuse = EAWDenoise_use_phi(denoising_m, c_phi_scale, n_phi_scale, p_phi_scale,
                                            stepWidth, denoise_iter, framedim_x, framedim_y, occ_map, total_diff_light, normal, pos_map)
        denoised_spec = EAWDenoise_use_phi(denoising_m, c_phi_scale, n_phi_scale, p_phi_scale,
                                            stepWidth, denoise_iter, framedim_x, framedim_y, occ_map, total_spec_light, normal, pos_map)
        denoised_indirect = EAWDenoise_use_phi_no_di(denoising_m, c_phi_scale, n_phi_scale, p_phi_scale,
                                            stepWidth, denoise_iter, framedim_x, framedim_y, occ_map, combined_color_indirect, normal, pos_map)
        
        denoised_indirect_diff = EAWDenoise_use_phi_no_di(denoising_m, c_phi_scale, n_phi_scale, p_phi_scale,
                                            stepWidth, denoise_iter, framedim_x, framedim_y, occ_map, total_diff_light_1, normal, pos_map)
        denoised_indirect_spec = EAWDenoise_use_phi_no_di(denoising_m, c_phi_scale, n_phi_scale, p_phi_scale,
                                            stepWidth, denoise_iter, framedim_x, framedim_y, occ_map, total_spec_light_1, normal, pos_map)
    else:
        factor = 2.0
        denoised_diffuse = bilateral_denoiser(framedim_y, framedim_x,
                                            torch.cat((total_diff_light, normal, gb_depth), dim=-1), factor)
        denoised_spec = bilateral_denoiser(framedim_y, framedim_x,
                                            torch.cat((total_spec_light, normal, gb_depth), dim=-1), factor)
        denoised_indirect = bilateral_denoiser_no_di(framedim_y, framedim_x,
                                            torch.cat((combined_color_indirect, normal, gb_depth), dim=-1), factor)
        
        denoised_indirect_diff = bilateral_denoiser_no_di(framedim_y, framedim_x,
                                            torch.cat((total_diff_light_1, normal, gb_depth), dim=-1), factor)
        denoised_indirect_spec = bilateral_denoiser_no_di(framedim_y, framedim_x,
                                            torch.cat((total_spec_light_1, normal, gb_depth), dim=-1), factor)

    diffuse = diffuse * (1.0 - roughnessSpecular[..., 1:2]) # kd * (1.0 - metalness)
    final_color = diffuse * denoised_diffuse + denoised_spec + denoised_indirect

    indices = torch.where(occ_map<=0.1)
    final_color[indices[0],:] = 1.0

    final_color = torch.nan_to_num(final_color, 0.0)

    return final_color, denoised_diffuse, denoised_spec, denoised_indirect, denoised_indirect_diff, denoised_indirect_spec