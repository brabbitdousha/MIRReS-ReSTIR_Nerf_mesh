import torch
import numpy as np

#H = 800*2
#W = 800*2

def InitialResampling_(m, LBVHNode_info, LBVHNode_aabb, vert, vert_ind, pos_map,
                       reservoirs,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir, pdf_, cdf_, mpdf_, mcdf_,
                       light_data, light_uv, light_inv_pdf
                       ):

    m.process_InitialResampling_(
    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind, pos_map=pos_map,
    reservoirs=reservoirs, env_tex=env_tex,
    env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y, frameIndex=frameIndex,
    occ_map=occ_map, normal_depth=normal_depth, brdf_map=brdf_map, ray_dir=ray_dir,
    pdf_=pdf_, cdf_=cdf_, mpdf_=mpdf_, mcdf_=mcdf_,
    light_data=light_data, light_uv=light_uv, light_inv_pdf=light_inv_pdf
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'

def TemporalResampling(m,
                       reservoirs, prev_reservoirs,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir,
                       prev_occ_map, prev_normal_depth, prev_brdf_map, prev_ray_dir,
                       motionVectors
                       ):

    m.process_TemporalResampling(reservoirs=reservoirs, prevReservoirs=prev_reservoirs,
    env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y, frameIndex=frameIndex,
    occ_map=occ_map, normal_depth=normal_depth, brdf_map=brdf_map, ray_dir=ray_dir,
    prev_occ_map=prev_occ_map, prev_normal_depth=prev_normal_depth, prev_brdf_map=prev_brdf_map, prev_ray_dir=prev_ray_dir,
    motionVectors=motionVectors
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'

def SpatialResampling_(m, LBVHNode_info, LBVHNode_aabb, vert, vert_ind, pos_map,
                       reservoirs, prev_reservoirs, neighborOffsets,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y, frameIndex,
                       occ_map, normal_depth, brdf_map, ray_dir
                       ):

    m.process_SpatialResampling_(
    g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind, pos_map=pos_map,
    reservoirs=reservoirs, prevReservoirs=prev_reservoirs, neighborOffsets=neighborOffsets,
    env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y, frameIndex=frameIndex,
    occ_map=occ_map, normal_depth=normal_depth, brdf_map=brdf_map, ray_dir=ray_dir
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'

def EvaluateFinalSamples(m,
                       reservoirs,
                       env_tex, env_width, env_height,
                       framedim_x, framedim_y,
                       finalSamples, vis_map, thr
                       ):

    m.process_EvaluateFinalSamples(reservoirs=reservoirs,
    env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y,
    finalSample=finalSamples, vis_map=vis_map, thr=thr
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'

def EvaluateFinalSamples_get_vis(m,
                       LBVHNode_info, LBVHNode_aabb, vert, vert_ind, pos_map,
                       reservoirs,
                       framedim_x, framedim_y,
                       vis_map
                       ):
    m.process_EvaluateFinalSamples_get_vis(g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind,
    reservoirs=reservoirs,
    framedim_x=framedim_x, framedim_y=framedim_y,
    pos_map=pos_map, vis_map=vis_map
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'


class EvaluateFinalSamples_di(torch.autograd.Function):
    @staticmethod
    def forward(ctx,    m,
                        res_light_data, res_light_pdf, res_M, res_weight,
                        env_tex, env_width, env_height,
                        framedim_x, framedim_y,
                        finalSamples_dir, finalSamples_distance, eva_vis_map
                        ):
        final_Li = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
        #final_Li = final_Li_[selected_index]
        m.process_EvaluateFinalSamples_di_(reservoirs=(res_light_data, res_light_pdf, res_M, res_weight),
        env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y,
        finalSample=(finalSamples_dir, finalSamples_distance, final_Li), vis_map=eva_vis_map
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        ctx.save_for_backward(res_light_data, res_light_pdf, res_M, res_weight,
                            env_tex,
                            finalSamples_dir, finalSamples_distance, final_Li, eva_vis_map)
        ctx.nums = [env_width, env_height, framedim_x, framedim_y]
        ctx.slang_m = m

        return final_Li
    
    @staticmethod
    def backward(ctx, grad_final_Li):

        grad_final_Li = grad_final_Li.contiguous()
        (res_light_data, res_light_pdf, res_M, res_weight,
                            env_tex,
                            finalSamples_dir, finalSamples_distance, final_Li, eva_vis_map) = ctx.saved_tensors
        env_width, env_height, framedim_x, framedim_y = ctx.nums
        m = ctx.slang_m
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #
        grad_env = torch.zeros_like(env_tex)

        m.process_EvaluateFinalSamples_di_.bwd(
        reservoirs=m.Reservoir(light_data=res_light_data, light_pdf=res_light_pdf, M=res_M, weight=res_weight),
        env_tex=(env_tex,grad_env), env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y,
        finalSample=m.FinalSample(dir=finalSamples_dir, distance=finalSamples_distance, Li=(final_Li, grad_final_Li)), vis_map=eva_vis_map
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return (None,
                        None, None, None, None,
                        grad_env, None, None,
                        None, None,
                        None, None, None)
                 
class FinalShading(torch.autograd.Function):
    @staticmethod
    def forward(ctx,    m,
                        finalSamples_dir,
                        finalSamples_distance,
                        finalSamples_Li,
                        env_tex, env_width, env_height,
                        framedim_x, framedim_y,
                        occ_map, normal, ray_dir,
                        diffuse_map, linearRoughness_specular_map
                        ):
        color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
        color_diff = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
        color_spec = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
        #color = color_[selected_index]
        #color_diff = color_diff_[selected_index]
        #color_spec = color_spec_[selected_index]
        m.process_FinalShading(finalSample=(finalSamples_dir, finalSamples_distance, finalSamples_Li),
        env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y,
        occ_map=occ_map, normal=normal, ray_dir=ray_dir,
        diffuse_map=diffuse_map, linearRoughness_specular_map=linearRoughness_specular_map, color=color,
        diff_light=color_diff, spec_light=color_spec
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        ctx.save_for_backward(finalSamples_dir, finalSamples_distance, finalSamples_Li,
                            env_tex, occ_map, normal, ray_dir, diffuse_map, linearRoughness_specular_map, color,
                            color_diff, color_spec)
        ctx.nums = [env_width, env_height, framedim_x, framedim_y]
        ctx.slang_m = m

        return color, color_diff, color_spec
    
    @staticmethod
    def backward(ctx, grad_color, grad_color_diff, grad_color_spec):

        grad_color = grad_color.contiguous()
        grad_color_diff = grad_color_diff.contiguous()
        grad_color_spec = grad_color_spec.contiguous()
        (finalSamples_dir, finalSamples_distance, finalSamples_Li, env_tex, occ_map, normal, 
         ray_dir, diffuse_map, linearRoughness_specular_map, color,
         color_diff, color_spec) = ctx.saved_tensors
        env_width, env_height, framedim_x, framedim_y = ctx.nums
        m = ctx.slang_m
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #

        grad_normal = torch.zeros_like(normal)
        grad_diffuse = torch.zeros_like(diffuse_map)
        grad_linearRoughness_specular = torch.zeros_like(linearRoughness_specular_map)
        grad_finalSamples_Li = torch.zeros_like(finalSamples_Li)

        m.process_FinalShading.bwd(finalSample=m.FinalSample(dir=finalSamples_dir, distance=finalSamples_distance, Li=(finalSamples_Li,grad_finalSamples_Li)),
        env_tex=env_tex, env_width=env_width, env_height=env_height, framedim_x=framedim_x, framedim_y=framedim_y,
        occ_map=occ_map, normal=(normal, grad_normal), ray_dir=ray_dir,
        diffuse_map=(diffuse_map, grad_diffuse), 
        linearRoughness_specular_map=(linearRoughness_specular_map,grad_linearRoughness_specular),
        color=(color, grad_color), diff_light=(color_diff, grad_color_diff), spec_light=(color_spec, grad_color_spec)
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return (None,   
                        None,
                        None,
                        grad_finalSamples_Li,
                        None, None, None,
                        None, None,
                        None, grad_normal, None,
                        grad_diffuse, grad_linearRoughness_specular)

def process_new_dir_for_pt(
                    m,
                    LBVHNode_info, LBVHNode_aabb, vert, vert_ind,
                    frameIndex, bounce_count, framedim_x, framedim_y, 
                     occ_map, pos_map, normal, ray_dir, prd,
                     diffuse_map, linearRoughness_specular_map,
                     new_pos_map, new_ray_d, new_occ_map, new_normal):
    
    m.process_new_dir_for_pt(
                          g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind,
                          frameIndex=frameIndex, bounce_count=bounce_count, framedim_x=framedim_x, framedim_y=framedim_y,
                          occ_map=occ_map, pos_map=pos_map, normal=normal, ray_dir=ray_dir, prd=prd,
                          diffuse_map=diffuse_map, linearRoughness_specular_map=linearRoughness_specular_map,
                          new_pos_map=new_pos_map, new_ray_d=new_ray_d, new_occ_map=new_occ_map, new_normal=new_normal)\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'
    
def indirect_one_hit(
                    m,
                    LBVHNode_info, LBVHNode_aabb, vert, vert_ind,
                    frameIndex, bounce_count, framedim_x, framedim_y, 
                     env_tex, env_width, env_height, pdf_, cdf_, mpdf_, mcdf_,
                     occ_map, pos_map, normal, ray_dir, prd,
                     diffuse_map, linearRoughness_specular_map, color, diff_color, spec_color,
                     new_pos_map, new_ray_d, new_occ_map, new_normal):
    
    m.process_path_tracing(
                          g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind,
                          frameIndex=frameIndex, bounce_count=bounce_count, framedim_x=framedim_x, framedim_y=framedim_y,
                          env_tex=env_tex, env_width=env_width, env_height=env_height,
                          pdf_=pdf_, cdf_=cdf_, mpdf_=mpdf_, mcdf_=mcdf_,
                          occ_map=occ_map, pos_map=pos_map, normal=normal, ray_dir=ray_dir, prd=prd,
                          diffuse_map=diffuse_map, linearRoughness_specular_map=linearRoughness_specular_map, color=color, diff_color=diff_color, spec_color=spec_color,
                          new_pos_map=new_pos_map, new_ray_d=new_ray_d, new_occ_map=new_occ_map, new_normal=new_normal)\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'

def indirect_one_hit_divided_no_grad(
                    m,
                    LBVHNode_info, LBVHNode_aabb, vert, vert_ind,
                    frameIndex, bounce_count, framedim_x, framedim_y, 
                     env_tex, env_width, env_height, pdf_, cdf_, mpdf_, mcdf_,
                     occ_map, pos_map, normal, ray_dir, prd,
                     diffuse_map, linearRoughness_specular_map, color, diff_color, spec_color,
                     new_pos_map, new_ray_d, new_occ_map, new_normal):
    
    m.process_path_tracing_divided_no_grad(
                          g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, vert=vert, v_indx=vert_ind,
                          frameIndex=frameIndex, bounce_count=bounce_count, framedim_x=framedim_x, framedim_y=framedim_y,
                          env_tex=env_tex, env_width=env_width, env_height=env_height,
                          pdf_=pdf_, cdf_=cdf_, mpdf_=mpdf_, mcdf_=mcdf_,
                          occ_map=occ_map, pos_map=pos_map, normal=normal, ray_dir=ray_dir, prd=prd,
                          diffuse_map=diffuse_map, linearRoughness_specular_map=linearRoughness_specular_map, color=color, diff_color=diff_color, spec_color=spec_color,
                          new_pos_map=new_pos_map, new_ray_d=new_ray_d, new_occ_map=new_occ_map, new_normal=new_normal)\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

    return 'hello'