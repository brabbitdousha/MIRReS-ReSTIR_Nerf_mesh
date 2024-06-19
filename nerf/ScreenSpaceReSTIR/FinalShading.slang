import utils.lightDi;
import utils.random;
import utils.helperDi;
import utils.res;
import utils.brdfDi;

#define MAX_Bounce 2
#define VIS_near 0.01f
#define F0 0.04f

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void process_FinalShading(FinalSample finalSample,
                          TensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y,
                          TensorView<float> occ_map, DiffTensorView<float> normal, TensorView<float> ray_dir,
                          DiffTensorView<float> diffuse_map, DiffTensorView<float> linearRoughness_specular_map, DiffTensorView<float> color,
                          DiffTensorView<float> diff_light, DiffTensorView<float> spec_light
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    float3 curr_normal = float3(normal[pixelIndex,0], normal[pixelIndex,1], normal[pixelIndex,2]);
    float3 curr_ray_dir = float3(no_diff ray_dir[pixelIndex,0], no_diff ray_dir[pixelIndex,1], no_diff ray_dir[pixelIndex,2]);
    float3 diffuse = float3(diffuse_map[pixelIndex,0], diffuse_map[pixelIndex,1], diffuse_map[pixelIndex,2]);
    float linearRoughness = linearRoughness_specular_map[pixelIndex, 0];

    float metallic = linearRoughness_specular_map[pixelIndex, 1];
    float3 specular = (float3(F0, F0, F0) * (1.0f - metallic) + diffuse * metallic);
    
    float specTrans = 0.f;
    float diffTrans = 0.f;
    float3 color_val = float3(0.f);
    float3 light_diffuse = float3(0.f);
    float3 light_spec = float3(0.f);

    if(occ_map[pixelIndex,0]>0.1f) 
    {
        //get final sample
        float3 dir = float3(no_diff finalSample.dir[pixelIndex, 0], no_diff finalSample.dir[pixelIndex, 1], no_diff finalSample.dir[pixelIndex, 2]);
        //float3 dir = float3(test_dir[pixelIndex, 0], test_dir[pixelIndex, 1], test_dir[pixelIndex, 2]);
        float distance = finalSample.distance[pixelIndex, 0];
        float3 Li = float3(finalSample.Li[pixelIndex, 0], finalSample.Li[pixelIndex, 1], finalSample.Li[pixelIndex, 2]);
        //float3 Li = float3(test_Li[pixelIndex, 0], test_Li[pixelIndex, 1], test_Li[pixelIndex, 2]);

        float3 diffuse_val = float3(0.f);
        float3 specular_val = float3(0.f);

        if(distance > 0.f)
        {
            //dir = float3(0.5f);//debug
            Frame surf_frame = create_frame(curr_normal);
            float3 wiLocal = -curr_ray_dir;
            frame_to_local(surf_frame, wiLocal);
            float3 woLocal = dir;
            frame_to_local(surf_frame, woLocal);

            float ROUGHNESS_THRESHOLD = 0.01;
            float kMinGGXAlpha = ROUGHNESS_THRESHOLD * ROUGHNESS_THRESHOLD;
            float alpha = linearRoughness * linearRoughness;
            if (alpha < kMinGGXAlpha) alpha = 0.f;

            float diffuseWeight = luminance(diffuse);
            float dielectricBSDF = (1.f - metallic) * (1.f - specTrans);
            float pDiffuseReflection = diffuseWeight * dielectricBSDF * (1.f - diffTrans);

            float metallicBRDF = metallic;
            float specularWeight = luminance(evalFresnelSchlick(specular, 1.f, dot(-curr_ray_dir, curr_normal)));
            float pSpecularReflection = specularWeight * (metallicBRDF + dielectricBSDF);

            float normFactor = pDiffuseReflection + pSpecularReflection;
            if (normFactor > 0.f)
            {
                normFactor = 1.f / normFactor;
                pDiffuseReflection *= normFactor;
                pSpecularReflection *= normFactor;
            }
            if (pDiffuseReflection > 0.f) diffuse_val = Diffuse_light(wiLocal, woLocal) * Li;
            if(pSpecularReflection > 0.f) specular_val = SpecularReflection_eval(wiLocal, woLocal, specular, alpha, true) * Li;
        }
        color_val += diffuse * (1.0f - metallic) * diffuse_val + specular_val;
        light_diffuse += diffuse_val;
        light_spec += specular_val;
    }
    else
    {
        //color_val = no_diff env_le(curr_ray_dir, env_tex, env_width, env_height);
        float3 temp_dir = curr_ray_dir;
        float3 r_dir = ngp_dir(temp_dir);
        color_val = no_diff env_le(r_dir, env_tex, env_width, env_height);
    }

    color[pixelIndex,0] = color_val.x;
    color[pixelIndex,1] = color_val.y;
    color[pixelIndex,2] = color_val.z;

    diff_light[pixelIndex, 0] = light_diffuse.x;
    diff_light[pixelIndex, 1] = light_diffuse.y;
    diff_light[pixelIndex, 2] = light_diffuse.z;

    spec_light[pixelIndex, 0] = light_spec.x;
    spec_light[pixelIndex, 1] = light_spec.y;
    spec_light[pixelIndex, 2] = light_spec.z;

}

[AutoPyBindCUDA]
[CUDAKernel]
void process_new_dir_for_pt(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                          TensorView<float> vert, TensorView<int> v_indx,
                          uint frameIndex, uint bounce_count, uint framedim_x, uint framedim_y,
                          TensorView<float> occ_map, TensorView<float> pos_map, TensorView<float> normal, TensorView<float> ray_dir, TensorView<float> prd,
                          TensorView<float> diffuse_map, TensorView<float> linearRoughness_specular_map,
                          TensorView<float> new_pos_map, TensorView<float> new_ray_d, TensorView<float> new_occ_map, TensorView<float> new_normal
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;

    float3 prd_throughput = float3(prd[pixelIndex, 0], prd[pixelIndex, 1], prd[pixelIndex, 2]);
    float specularBounce = prd[pixelIndex, 3];
    float is_stop = prd[pixelIndex, 4];
    new_occ_map[pixelIndex, 0] = 0.f;
    prd[pixelIndex, 4] = 1.f;

    if (bounce_count == 0)
    {
        prd_throughput = float3(1.0f);
        specularBounce = 0.f;
        is_stop = 0.f;
        prd[pixelIndex, 0] = prd_throughput.x;
        prd[pixelIndex, 1] = prd_throughput.y;
        prd[pixelIndex, 2] = prd_throughput.z;

        prd[pixelIndex, 3] = specularBounce;
    }
    
    if (is_stop > 0.f)
    {
        return;
    }

    float3 new_pos = float3(0.f);
    float new_occ = 0.f;
    float3 new_rayd = float3(0.f);

    float3 curr_normal = float3(normal[pixelIndex,0], normal[pixelIndex,1], normal[pixelIndex,2]);
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_surf_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);
    float3 diffuse = float3(diffuse_map[pixelIndex,0], diffuse_map[pixelIndex,1], diffuse_map[pixelIndex,2]);
    float linearRoughness = linearRoughness_specular_map[pixelIndex, 0];
    float metallic = linearRoughness_specular_map[pixelIndex, 1];
    float3 specular = (float3(F0, F0, F0) * (1.0f - metallic) + diffuse * metallic);
    float specTrans = 0.f;
    float diffTrans = 0.f;
    uint sg = Seed_Generator(pixel, frameIndex);

    if(occ_map[pixelIndex,0]>0.1f) 
    {
        // set up first for brdf------------
        float ROUGHNESS_THRESHOLD = 0.01;
        float kMinGGXAlpha = ROUGHNESS_THRESHOLD * ROUGHNESS_THRESHOLD;
        float alpha = linearRoughness * linearRoughness;
        if (alpha < kMinGGXAlpha) alpha = 0.f;

        float diffuseWeight = luminance(diffuse);
        float dielectricBSDF = (1.f - metallic) * (1.f - specTrans);
        float pDiffuseReflection = diffuseWeight * dielectricBSDF * (1.f - diffTrans);

        float metallicBRDF = metallic;
        float specularWeight = luminance(evalFresnelSchlick(specular, 1.f, dot(-curr_ray_dir, curr_normal)));
        float pSpecularReflection = specularWeight * (metallicBRDF + dielectricBSDF);

        float normFactor = pDiffuseReflection + pSpecularReflection;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pSpecularReflection *= normFactor;
        }
        Frame surf_frame = create_frame(curr_normal);
        float3 wiLocal = -curr_ray_dir;
        frame_to_local(surf_frame, wiLocal);

        uint sampledSpecular = 0;
        // let's get a new direction!
        float3 out_dir;
        float out_pdf;
        float3 out_weight = float3(1.0f);
        float3 diffuse_col = diffuse * (1.0f - metallic);
        bool valid = FalcorBRDF_sample(pDiffuseReflection, pSpecularReflection,
                                       wiLocal, out_dir, out_pdf, sampledSpecular, out_weight, sg,
                                       alpha, specular, diffuse_col, true, false);
        if (valid)
        {
            if (IsBlack(out_weight) || out_pdf == 0.f)
            {
                is_stop = 1.f;
                prd[pixelIndex, 4] = is_stop;

            }
            else if (bounce_count+1<=MAX_Bounce)
            {
                frame_to_global(surf_frame, out_dir);
                out_dir = normalize(out_dir);
                float3 bounce_pos = curr_surf_pos;
                float3 light_dir = out_dir;
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 t_pos = float3(0.f);
                float3 hit_normal = float3(1.f);
                float vis_near = VIS_near;
                bounce_pos = bounce_pos + vis_near * light_dir;
                bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                     bounce_pos, light_dir, t_min, t_max, t_hit, t_pos, hit_normal);
                
                specularBounce = float(sampledSpecular);
                prd_throughput *= out_weight;
                new_rayd = out_dir;
                new_pos = t_pos;
                
                //write for next bounce
                prd[pixelIndex, 0] = prd_throughput.x;
                prd[pixelIndex, 1] = prd_throughput.y;
                prd[pixelIndex, 2] = prd_throughput.z;

                prd[pixelIndex, 3] = specularBounce;

                new_ray_d[pixelIndex, 0] = new_rayd.x;
                new_ray_d[pixelIndex, 1] = new_rayd.y;
                new_ray_d[pixelIndex, 2] = new_rayd.z;
                
                if(hit)
                {
                    prd[pixelIndex, 4] = 0.f;//no stop

                    new_pos_map[pixelIndex, 0] = new_pos.x;
                    new_pos_map[pixelIndex, 1] = new_pos.y;
                    new_pos_map[pixelIndex, 2] = new_pos.z;

                    new_normal[pixelIndex, 0] = hit_normal.x;
                    new_normal[pixelIndex, 1] = hit_normal.y;
                    new_normal[pixelIndex, 2] = hit_normal.z;
                    
                    new_occ_map[pixelIndex, 0] = 1.f;
                }
                else if(specularBounce>0.f)
                {
                    prd[pixelIndex, 4] = 0.f; //specular bounce can get envmap value in next bounce
                }

            }
        }
    }

}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void process_path_tracing(
                          TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                          TensorView<float> vert, TensorView<int> v_indx,
                          uint frameIndex, uint bounce_count, uint framedim_x, uint framedim_y,
                          DiffTensorView<float> env_tex, TensorView<float> env_tex_no_di, uint env_width, uint env_height,
                          TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_,
                          TensorView<float> occ_map, TensorView<float> pos_map, DiffTensorView<float> normal, TensorView<float> ray_dir, TensorView<float> prd,
                          DiffTensorView<float> diffuse_map, DiffTensorView<float> linearRoughness_specular_map, DiffTensorView<float> color, DiffTensorView<float> diff_color, DiffTensorView<float> spec_color,
                          TensorView<float> new_pos_map, TensorView<float> new_ray_d, TensorView<float> new_occ_map, TensorView<float> new_normal
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    float3 prd_throughput = float3(no_diff prd[pixelIndex, 0], no_diff prd[pixelIndex, 1], no_diff prd[pixelIndex, 2]);
    float specularBounce = prd[pixelIndex, 3];
    float is_stop = prd[pixelIndex, 4];
    new_occ_map[pixelIndex, 0] = 0.f;
    prd[pixelIndex, 4] = 1.f;

    if (bounce_count == 0)
    {
        prd_throughput = float3(1.0f);
        specularBounce = 0.f;
        is_stop = 0.f;
        prd[pixelIndex, 0] = prd_throughput.x;
        prd[pixelIndex, 1] = prd_throughput.y;
        prd[pixelIndex, 2] = prd_throughput.z;

        prd[pixelIndex, 3] = specularBounce;
    }
    
    if (is_stop > 0.f)
    {
        color[pixelIndex, 0] = 0.f;
        color[pixelIndex, 1] = 0.f;
        color[pixelIndex, 2] = 0.f;

        diff_color[pixelIndex, 0] = 0.f;
        diff_color[pixelIndex, 1] = 0.f;
        diff_color[pixelIndex, 2] = 0.f;

        spec_color[pixelIndex, 0] = 0.f;
        spec_color[pixelIndex, 1] = 0.f;
        spec_color[pixelIndex, 2] = 0.f;

        return;
    }
    float3 new_pos = float3(0.f);
    float new_occ = 0.f;
    float3 new_rayd = float3(0.f);

    float3 curr_normal = float3(normal[pixelIndex, 0], normal[pixelIndex, 1], normal[pixelIndex, 2]);
    float3 curr_ray_dir = float3(no_diff ray_dir[pixelIndex, 0], no_diff ray_dir[pixelIndex, 1], no_diff ray_dir[pixelIndex, 2]);
    float3 curr_surf_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);
    float3 diffuse = float3(diffuse_map[pixelIndex, 0], diffuse_map[pixelIndex, 1], diffuse_map[pixelIndex, 2]);
    float linearRoughness = linearRoughness_specular_map[pixelIndex, 0];
    float metallic = linearRoughness_specular_map[pixelIndex, 1];
    float3 specular = (float3(F0, F0, F0) * (1.0f - metallic) + diffuse * metallic);
    float specTrans = 0.f;
    float diffTrans = 0.f;
    float3 color_val = float3(0.f);
    float3 diff_color_val = float3(0.f);
    float3 spec_color_val = float3(0.f);
    uint sg = Seed_Generator(pixel, frameIndex);

    if (occ_map[pixelIndex, 0] > 0.1f)
    {
        float3 diffuse_val = float3(0.f);
        float3 specular_val = float3(0.f);

        // set up first for brdf------------
        float ROUGHNESS_THRESHOLD = 0.01;
        float kMinGGXAlpha = ROUGHNESS_THRESHOLD * ROUGHNESS_THRESHOLD;
        float alpha = linearRoughness * linearRoughness;
        if (alpha < kMinGGXAlpha) alpha = 0.f;

        float diffuseWeight = luminance(diffuse);
        float dielectricBSDF = (1.f - metallic) * (1.f - specTrans);
        float pDiffuseReflection = diffuseWeight * dielectricBSDF * (1.f - diffTrans);

        float metallicBRDF = metallic;
        float specularWeight = luminance(evalFresnelSchlick(specular, 1.f, dot(-curr_ray_dir, curr_normal)));
        float pSpecularReflection = specularWeight * (metallicBRDF + dielectricBSDF);

        float normFactor = pDiffuseReflection + pSpecularReflection;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pSpecularReflection *= normFactor;
        }
        //-------------------------------
        // Sample light source with multiple importance sampling
        float lightPdf = 0.0f;
        float scatteringPdf = 0.0f;
        LightSample samp = no_diff new_light_sample();
        float3 samp_dir;
        float3 samp_weight;
        float samp_pdf;
        float2 light_uv;
        bool res = InfiniteAreaLight_Sample_Li_no_env(env_width, env_height, float2(sampleNext1D(sg), sampleNext1D(sg)),
                                               pdf_, cdf_, mpdf_, mcdf_,
                                               samp_dir, samp_pdf, light_uv);

        if (res)
        {
            samp.light_data.x = 1;
            samp.dir = detach(samp_dir);
            samp.pdf = detach(samp_pdf);
            //samp_weight = scene_env_tex_sample_di(env_tex, light_uv, env_width, env_height)/samp_pdf;
            float3 temp_dir = detach(samp_dir);
            float3 r_dir = ngp_dir(temp_dir);
            samp_weight = env_le_di(r_dir, env_tex, env_width, env_height) / samp_pdf;
            samp.weight = samp_weight;
        }

        Frame surf_frame = create_frame(curr_normal);
        float3 wiLocal = -curr_ray_dir;
        frame_to_local(surf_frame, wiLocal);

        lightPdf = samp.pdf;
        float3 Li = samp.weight;
        if ((no_diff is_valid_light_sample(samp)) && lightPdf > 0 && !(IsBlack(Li)))
        {
            float3 diff_f = float3(0.0, 0.0, 0.0);
            float3 spec_f = float3(0.0, 0.0, 0.0);
            float3 total_f = float3(0.0, 0.0, 0.0);
            float3 woLocal = samp.dir;
            frame_to_local(surf_frame, woLocal);
            if (!IsBlack(curr_normal))
            {
                if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wiLocal, woLocal);
                if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wiLocal, woLocal, specular, alpha, true);
                total_f  = diffuse*diff_f + spec_f;

                scatteringPdf = FalcorBRDF_evalPdf(pDiffuseReflection, pSpecularReflection, wiLocal, woLocal, alpha, true, false);
            }

            if (!IsBlack(total_f))
            {
                // Compute effect of visibility for light source sample
                float3 shadow_pos = curr_surf_pos;
                float3 light_dir = normalize(samp.dir);
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 t_pos = float3(0.f);
                float vis_near = VIS_near;
                shadow_pos = shadow_pos + vis_near * light_dir;
                bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                    shadow_pos, light_dir, t_min, t_max, t_hit, t_pos);
                float3 tr = float3(1.0, 1.0, 1.0);
                if(hit) tr = float3(0.0,0.0,0.0);
                Li *= tr;

                // Add light's contribution to reflected radiance
                if (!IsBlack(Li))
                {
                    float mis_weight = 1.0;
                    // if (!is_delta_light(samp.light_data))
                    //{
                    mis_weight = no_diff power_heuristic(lightPdf, scatteringPdf);
                    //}

                    float3 value = prd_throughput * total_f * Li * mis_weight;
                    float3 diff_value = prd_throughput * diff_f * Li * mis_weight;
                    float3 spec_value = prd_throughput * spec_f * Li * mis_weight;

                    // vec3 value = prd.throughput * 1.0 * 0.5*0.5;
                    color_val += value;
                    diff_color_val += diff_value;
                    spec_color_val += spec_value;
                }
            }
        }

        // Sample BSDF with multiple importance sampling
        // if (!is_delta_light(samp.light_data))
        //{
        float3 f = float3(0.0f);
        float3 diff_f = float3(0.0f);
        float3 spec_f = float3(0.0f);

        uint sampledSpecular = 0;
        float3 m_wi;
        float m_pdf;
        if (!IsBlack(curr_normal))
        {
            // Sample scattered direction for surface interactions
            float3 bsdf_weight = float3(1.0f);
            float3 bsdf_diff_weight = float3(1.0f);
            float3 bsdf_spec_weight = float3(1.0f);

            bool valid = FalcorBRDF_sample_no_weight(pDiffuseReflection, pSpecularReflection,
                                           wiLocal, m_wi, m_pdf, sampledSpecular, sg,
                                           alpha, specular, diffuse, true, false);
            if (valid)
            {
                if (pDiffuseReflection > 0.f) bsdf_diff_weight = Diffuse_light(wiLocal, m_wi);
                if (pSpecularReflection > 0.f) bsdf_spec_weight = SpecularReflection_eval(wiLocal, m_wi, specular, alpha, true);
                bsdf_weight  = diffuse*bsdf_diff_weight + bsdf_spec_weight;

                frame_to_global(surf_frame, m_wi);
                scatteringPdf = m_pdf;
                f = bsdf_weight/m_pdf;
                diff_f = bsdf_diff_weight/m_pdf;
                spec_f = bsdf_spec_weight/m_pdf;

                f *= m_pdf;
                diff_f *= m_pdf;
                spec_f *= m_pdf;

                float3 m_safe_wi = normalize(m_wi);

                if (!IsBlack(f) && scatteringPdf > 0)
                {
                    float weight = 1.0f;
                    bool islightpdfZero = false; // ugly
                    if (sampledSpecular==0)
                    {
                        float pdf_li = no_diff sample_light_pdf(m_safe_wi, env_width, env_height, pdf_, mpdf_);;
                        lightPdf = pdf_li;
                        if (lightPdf == 0.0f) { islightpdfZero = true;}
                        weight = no_diff power_heuristic(scatteringPdf, lightPdf);
                    }
                    // Find intersection and compute transmittance
                    float3 shadow_pos = curr_surf_pos;
                    float3 light_dir = m_safe_wi;
                    float t_min = 0.f;
                    float t_max = 1e7;
                    float t_hit = 0.f;
                    float3 t_pos = float3(0.f);
                    float vis_near = VIS_near;
                    shadow_pos = shadow_pos + vis_near * light_dir;
                    bool foundSurfaceInteraction = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                       shadow_pos, light_dir, t_min, t_max, t_hit, t_pos);
                    //bool foundSurfaceInteraction = false;
                    float3 Tr = float3(1.0, 1.0, 1.0);
                    Li = float3(0.f);
                    if (!foundSurfaceInteraction)
                    {
                        //Li = env_le_di(m_safe_wi, env_tex, env_width, env_height);
                        float3 temp_dir = m_safe_wi;
                        float3 r_dir = ngp_dir(temp_dir);
                        Li = env_le_di(r_dir, env_tex, env_width, env_height);
                    }

                    if (!IsBlack(Li) && !islightpdfZero)
                    { 
                        float3 value_bsdf = prd_throughput * f * Li * Tr * weight / scatteringPdf;
                        float3 value_diff_bsdf = prd_throughput * diff_f * Li * Tr * weight / scatteringPdf;
                        float3 value_spec_bsdf = prd_throughput * spec_f * Li * Tr * weight / scatteringPdf;

                        color_val += value_bsdf;
                        diff_color_val += value_diff_bsdf;
                        spec_color_val += value_spec_bsdf;
                    }

                }
            }
        }

        // let's get a new direction!
        float3 out_dir;
        float out_pdf;
        float3 out_weight = float3(1.0f);
        bool valid = FalcorBRDF_sample(pDiffuseReflection, pSpecularReflection,
                                       wiLocal, out_dir, out_pdf, sampledSpecular, out_weight, sg,
                                       alpha, specular, diffuse, true, false);
        if (valid)
        {
            if (IsBlack(out_weight) || out_pdf == 0.f)
            {
                is_stop = 1.f;
                prd[pixelIndex, 4] = is_stop;

            }
            else if (bounce_count+1<=MAX_Bounce)
            {
                frame_to_global(surf_frame, out_dir);
                out_dir = normalize(out_dir);
                float3 bounce_pos = curr_surf_pos;
                float3 light_dir = out_dir;
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 t_pos = float3(0.f);
                float3 hit_normal = float3(1.f);
                float vis_near = VIS_near;
                bounce_pos = bounce_pos + vis_near * light_dir;
                bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                     bounce_pos, light_dir, t_min, t_max, t_hit, t_pos, hit_normal);
                
                specularBounce = float(sampledSpecular);
                prd_throughput *= out_weight;
                new_rayd = out_dir;
                new_pos = t_pos;
                
                //write for next bounce
                prd[pixelIndex, 0] = prd_throughput.x;
                prd[pixelIndex, 1] = prd_throughput.y;
                prd[pixelIndex, 2] = prd_throughput.z;

                prd[pixelIndex, 3] = specularBounce;

                new_ray_d[pixelIndex, 0] = new_rayd.x;
                new_ray_d[pixelIndex, 1] = new_rayd.y;
                new_ray_d[pixelIndex, 2] = new_rayd.z;
                
                if(hit)
                {
                    prd[pixelIndex, 4] = 0.f;//no stop

                    new_pos_map[pixelIndex, 0] = new_pos.x;
                    new_pos_map[pixelIndex, 1] = new_pos.y;
                    new_pos_map[pixelIndex, 2] = new_pos.z;

                    new_normal[pixelIndex, 0] = hit_normal.x;
                    new_normal[pixelIndex, 1] = hit_normal.y;
                    new_normal[pixelIndex, 2] = hit_normal.z;
                    
                    new_occ_map[pixelIndex, 0] = 1.f;
                }
                else if(specularBounce>0.f)
                {
                    prd[pixelIndex, 4] = 0.f; //specular bounce can get envmap value in next bounce
                }

            }
        }
    }
    else
    {
        if (bounce_count == 0)
        {
            //color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
            float3 temp_dir = curr_ray_dir;
            float3 r_dir = ngp_dir(temp_dir);
            color_val += prd_throughput * no_diff env_le(r_dir, env_tex_no_di, env_width, env_height);
        }
        else if (specularBounce > 0.f)
        {
            float3 temp_dir = curr_ray_dir;
            float3 r_dir = ngp_dir(temp_dir);
            color_val += prd_throughput * env_le_di(r_dir, env_tex, env_width, env_height);
            spec_color_val += prd_throughput * env_le_di(r_dir, env_tex, env_width, env_height);
            //color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
            //spec_color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
        }
        prd[pixelIndex, 4] = 1.f;
    }

    color[pixelIndex, 0] = color_val.x;
    color[pixelIndex, 1] = color_val.y;
    color[pixelIndex, 2] = color_val.z;

    diff_color[pixelIndex, 0] = diff_color_val.x;
    diff_color[pixelIndex, 1] = diff_color_val.y;
    diff_color[pixelIndex, 2] = diff_color_val.z;

    spec_color[pixelIndex, 0] = spec_color_val.x;
    spec_color[pixelIndex, 1] = spec_color_val.y;
    spec_color[pixelIndex, 2] = spec_color_val.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_path_tracing_divided_no_grad(
    TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
    TensorView<float> vert, TensorView<int> v_indx,
    uint frameIndex, uint bounce_count, uint framedim_x, uint framedim_y,
    TensorView<float> env_tex, uint env_width, uint env_height,
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_,
    TensorView<float> occ_map, TensorView<float> pos_map, TensorView<float> normal, TensorView<float> ray_dir, TensorView<float> prd,
    TensorView<float> diffuse_map, TensorView<float> linearRoughness_specular_map, TensorView<float> color, TensorView<float> diff_color, TensorView<float> spec_color,
    TensorView<float> new_pos_map, TensorView<float> new_ray_d, TensorView<float> new_occ_map, TensorView<float> new_normal
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    float3 prd_throughput = float3(prd[pixelIndex, 0], prd[pixelIndex, 1], prd[pixelIndex, 2]);
    float specularBounce = prd[pixelIndex, 3];
    float is_stop = prd[pixelIndex, 4];
    new_occ_map[pixelIndex, 0] = 0.f;
    prd[pixelIndex, 4] = 1.f;

    if (bounce_count == 0)
    {
        prd_throughput = float3(1.0f);
        specularBounce = 0.f;
        is_stop = 0.f;
        prd[pixelIndex, 0] = prd_throughput.x;
        prd[pixelIndex, 1] = prd_throughput.y;
        prd[pixelIndex, 2] = prd_throughput.z;

        prd[pixelIndex, 3] = specularBounce;
    }

    if (is_stop > 0.f)
    {
        color[pixelIndex, 0] = 0.f;
        color[pixelIndex, 1] = 0.f;
        color[pixelIndex, 2] = 0.f;

        diff_color[pixelIndex, 0] = 0.f;
        diff_color[pixelIndex, 1] = 0.f;
        diff_color[pixelIndex, 2] = 0.f;

        spec_color[pixelIndex, 0] = 0.f;
        spec_color[pixelIndex, 1] = 0.f;
        spec_color[pixelIndex, 2] = 0.f;

        return;
    }
    float3 new_pos = float3(0.f);
    float new_occ = 0.f;
    float3 new_rayd = float3(0.f);

    float3 curr_normal = float3(normal[pixelIndex, 0], normal[pixelIndex, 1], normal[pixelIndex, 2]);
    float3 curr_ray_dir = float3(ray_dir[pixelIndex, 0], ray_dir[pixelIndex, 1], ray_dir[pixelIndex, 2]);
    float3 curr_surf_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);
    float3 diffuse = float3(diffuse_map[pixelIndex, 0], diffuse_map[pixelIndex, 1], diffuse_map[pixelIndex, 2]);
    float linearRoughness = linearRoughness_specular_map[pixelIndex, 0];
    float metallic = linearRoughness_specular_map[pixelIndex, 1];
    float3 specular = (float3(F0, F0, F0) * (1.0f - metallic) + diffuse * metallic);
    float specTrans = 0.f;
    float diffTrans = 0.f;
    float3 color_val = float3(0.f);
    float3 diff_color_val = float3(0.f);
    float3 spec_color_val = float3(0.f);
    uint sg = Seed_Generator(pixel, frameIndex);

    if (occ_map[pixelIndex, 0] > 0.1f)
    {
        float3 diffuse_val = float3(0.f);
        float3 specular_val = float3(0.f);

        // set up first for brdf------------
        float ROUGHNESS_THRESHOLD = 0.01;
        float kMinGGXAlpha = ROUGHNESS_THRESHOLD * ROUGHNESS_THRESHOLD;
        float alpha = linearRoughness * linearRoughness;
        if (alpha < kMinGGXAlpha) alpha = 0.f;

        float diffuseWeight = luminance(diffuse);
        float dielectricBSDF = (1.f - metallic) * (1.f - specTrans);
        float pDiffuseReflection = diffuseWeight * dielectricBSDF * (1.f - diffTrans);

        float metallicBRDF = metallic;
        float specularWeight = luminance(evalFresnelSchlick(specular, 1.f, dot(-curr_ray_dir, curr_normal)));
        float pSpecularReflection = specularWeight * (metallicBRDF + dielectricBSDF);

        float normFactor = pDiffuseReflection + pSpecularReflection;
        if (normFactor > 0.f)
        {
            normFactor = 1.f / normFactor;
            pDiffuseReflection *= normFactor;
            pSpecularReflection *= normFactor;
        }
        //-------------------------------
        // Sample light source with multiple importance sampling
        float lightPdf = 0.0f;
        float scatteringPdf = 0.0f;
        LightSample samp = new_light_sample();
        float3 samp_dir;
        float3 samp_weight;
        float samp_pdf;
        float2 light_uv;
        bool res = InfiniteAreaLight_Sample_Li_no_env(env_width, env_height, float2(sampleNext1D(sg), sampleNext1D(sg)),
                                                      pdf_, cdf_, mpdf_, mcdf_,
                                                      samp_dir, samp_pdf, light_uv);

        if (res)
        {
            samp.light_data.x = 1;
            samp.dir = detach(samp_dir);
            samp.pdf = detach(samp_pdf);
            //samp_weight = scene_env_tex_sample(env_tex, light_uv, env_width, env_height) / samp_pdf;
            float3 temp_dir = detach(samp_dir);
            float3 r_dir = ngp_dir(temp_dir);
            samp_weight = env_le(r_dir, env_tex, env_width, env_height) / samp_pdf;

            samp.weight = samp_weight;
        }

        Frame surf_frame = create_frame(curr_normal);
        float3 wiLocal = -curr_ray_dir;
        frame_to_local(surf_frame, wiLocal);

        lightPdf = samp.pdf;
        float3 Li = samp.weight;
        if ((is_valid_light_sample(samp)) && lightPdf > 0 && !(IsBlack(Li)))
        {
            float3 diff_f = float3(0.0, 0.0, 0.0);
            float3 spec_f = float3(0.0, 0.0, 0.0);
            float3 total_f = float3(0.0, 0.0, 0.0);
            float3 woLocal = samp.dir;
            frame_to_local(surf_frame, woLocal);
            if (!IsBlack(curr_normal))
            {
                if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wiLocal, woLocal);
                if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wiLocal, woLocal, specular, alpha, true);
                float3 diffuse_col = diffuse * (1.0f - metallic);
                total_f = diffuse_col * diff_f + spec_f;
                diff_f = diffuse_col * diff_f;

                scatteringPdf = FalcorBRDF_evalPdf(pDiffuseReflection, pSpecularReflection, wiLocal, woLocal, alpha, true, false);
            }

            if (!IsBlack(total_f))
            {
                // Compute effect of visibility for light source sample
                float3 shadow_pos = curr_surf_pos;
                float3 light_dir = normalize(samp.dir);
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 t_pos = float3(0.f);
                float vis_near = VIS_near;
                shadow_pos = shadow_pos + vis_near * light_dir;
                bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                   shadow_pos, light_dir, t_min, t_max, t_hit, t_pos);
                float3 tr = float3(1.0, 1.0, 1.0);
                if (hit) tr = float3(0.0, 0.0, 0.0);
                Li *= tr;

                // Add light's contribution to reflected radiance
                if (!IsBlack(Li))
                {
                    float mis_weight = 1.0;
                    // if (!is_delta_light(samp.light_data))
                    //{
                    mis_weight = power_heuristic(lightPdf, scatteringPdf);
                    //}

                    float3 value = prd_throughput * total_f * Li * mis_weight;
                    float3 diff_value = prd_throughput * diff_f * Li * mis_weight;
                    float3 spec_value = prd_throughput * spec_f * Li * mis_weight;

                    // vec3 value = prd.throughput * 1.0 * 0.5*0.5;
                    color_val += value;
                    diff_color_val += diff_value;
                    spec_color_val += spec_value;
                }
            }
        }

        // Sample BSDF with multiple importance sampling
        // if (!is_delta_light(samp.light_data))
        //{
        float3 f = float3(0.0f);
        float3 diff_f = float3(0.0f);
        float3 spec_f = float3(0.0f);

        uint sampledSpecular = 0;
        float3 m_wi;
        float m_pdf;
        if (!IsBlack(curr_normal))
        {
            // Sample scattered direction for surface interactions
            float3 bsdf_weight = float3(1.0f);
            float3 bsdf_diff_weight = float3(1.0f);
            float3 bsdf_spec_weight = float3(1.0f);

            float3 diffuse_col = diffuse * (1.0f - metallic);
            bool valid = FalcorBRDF_sample_no_weight(pDiffuseReflection, pSpecularReflection,
                                                     wiLocal, m_wi, m_pdf, sampledSpecular, sg,
                                                     alpha, specular, diffuse_col, true, false);
            if (valid)
            {
                if (pDiffuseReflection > 0.f) bsdf_diff_weight = Diffuse_light(wiLocal, m_wi);
                if (pSpecularReflection > 0.f) bsdf_spec_weight = SpecularReflection_eval(wiLocal, m_wi, specular, alpha, true);
                bsdf_weight = diffuse_col * bsdf_diff_weight + bsdf_spec_weight;

                frame_to_global(surf_frame, m_wi);
                scatteringPdf = m_pdf;
                f = bsdf_weight / m_pdf;
                diff_f = diffuse_col * bsdf_diff_weight / m_pdf;
                spec_f = bsdf_spec_weight / m_pdf;

                f *= m_pdf;
                diff_f *= m_pdf;
                spec_f *= m_pdf;

                float3 m_safe_wi = normalize(m_wi);

                if (!IsBlack(f) && scatteringPdf > 0)
                {
                    float weight = 1.0f;
                    bool islightpdfZero = false; // ugly
                    if (sampledSpecular == 0)
                    {
                        float pdf_li = sample_light_pdf(m_safe_wi, env_width, env_height, pdf_, mpdf_); ;
                        lightPdf = pdf_li;
                        if (lightPdf == 0.0f) { islightpdfZero = true;}
                        weight = power_heuristic(scatteringPdf, lightPdf);
                    }
                    // Find intersection and compute transmittance
                    float3 shadow_pos = curr_surf_pos;
                    float3 light_dir = m_safe_wi;
                    float t_min = 0.f;
                    float t_max = 1e7;
                    float t_hit = 0.f;
                    float3 t_pos = float3(0.f);
                    float vis_near = VIS_near;
                    shadow_pos = shadow_pos + vis_near * light_dir;
                    bool foundSurfaceInteraction = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                                           shadow_pos, light_dir, t_min, t_max, t_hit, t_pos);
                    // bool foundSurfaceInteraction = false;
                    float3 Tr = float3(1.0, 1.0, 1.0);
                    Li = float3(0.f);
                    if (!foundSurfaceInteraction)
                    {
                        //Li = env_le(m_safe_wi, env_tex, env_width, env_height);
                        float3 temp_dir = m_safe_wi;
                        float3 r_dir = ngp_dir(temp_dir);
                        Li = env_le(r_dir, env_tex, env_width, env_height);
                    }

                    if (!IsBlack(Li) && !islightpdfZero)
                    {
                        float3 value_bsdf = prd_throughput * f * Li * Tr * weight / scatteringPdf;
                        float3 value_diff_bsdf = prd_throughput * diff_f * Li * Tr * weight / scatteringPdf;
                        float3 value_spec_bsdf = prd_throughput * spec_f * Li * Tr * weight / scatteringPdf;

                        color_val += value_bsdf;
                        diff_color_val += value_diff_bsdf;
                        spec_color_val += value_spec_bsdf;
                    }
                }
            }
        }

        // let's get a new direction!
        float3 out_dir;
        float out_pdf;
        float3 out_weight = float3(1.0f);
        float3 diffuse_col = diffuse * (1.0f - metallic);
        bool valid = FalcorBRDF_sample(pDiffuseReflection, pSpecularReflection,
                                       wiLocal, out_dir, out_pdf, sampledSpecular, out_weight, sg,
                                       alpha, specular, diffuse_col, true, false);
        if (valid)
        {
            if (IsBlack(out_weight) || out_pdf == 0.f)
            {
                is_stop = 1.f;
                prd[pixelIndex, 4] = is_stop;
            }
            else if (bounce_count + 1 <= MAX_Bounce)
            {
                frame_to_global(surf_frame, out_dir);
                out_dir = normalize(out_dir);
                float3 bounce_pos = curr_surf_pos;
                float3 light_dir = out_dir;
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 t_pos = float3(0.f);
                float3 hit_normal = float3(1.f);
                float vis_near = VIS_near;
                bounce_pos = bounce_pos + vis_near * light_dir;
                bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                                               bounce_pos, light_dir, t_min, t_max, t_hit, t_pos, hit_normal);

                specularBounce = float(sampledSpecular);
                prd_throughput *= out_weight;
                new_rayd = out_dir;
                new_pos = t_pos;

                // write for next bounce
                prd[pixelIndex, 0] = prd_throughput.x;
                prd[pixelIndex, 1] = prd_throughput.y;
                prd[pixelIndex, 2] = prd_throughput.z;

                prd[pixelIndex, 3] = specularBounce;

                new_ray_d[pixelIndex, 0] = new_rayd.x;
                new_ray_d[pixelIndex, 1] = new_rayd.y;
                new_ray_d[pixelIndex, 2] = new_rayd.z;

                if (hit)
                {
                    prd[pixelIndex, 4] = 0.f; // no stop

                    new_pos_map[pixelIndex, 0] = new_pos.x;
                    new_pos_map[pixelIndex, 1] = new_pos.y;
                    new_pos_map[pixelIndex, 2] = new_pos.z;

                    new_normal[pixelIndex, 0] = hit_normal.x;
                    new_normal[pixelIndex, 1] = hit_normal.y;
                    new_normal[pixelIndex, 2] = hit_normal.z;

                    new_occ_map[pixelIndex, 0] = 1.f;
                }
                else if (specularBounce > 0.f)
                {
                    prd[pixelIndex, 4] = 0.f; // specular bounce can get envmap value in next bounce
                }
            }
        }
    }
    else
    {
        if (bounce_count == 0)
        {
            //color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
            float3 temp_dir = curr_ray_dir;
            float3 r_dir = ngp_dir(temp_dir);
            color_val += prd_throughput * env_le(r_dir, env_tex, env_width, env_height);
        }
        else if (specularBounce > 0.f)
        {
            float3 temp_dir = curr_ray_dir;
            float3 r_dir = ngp_dir(temp_dir);
            color_val += prd_throughput * env_le(r_dir, env_tex, env_width, env_height);
            spec_color_val += prd_throughput * env_le(r_dir, env_tex, env_width, env_height);
            //color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
            //spec_color_val += prd_throughput * env_le(curr_ray_dir, env_tex, env_width, env_height);
        }
        prd[pixelIndex, 4] = 1.f;
    }

    color[pixelIndex, 0] = color_val.x;
    color[pixelIndex, 1] = color_val.y;
    color[pixelIndex, 2] = color_val.z;

    diff_color[pixelIndex, 0] = diff_color_val.x;
    diff_color[pixelIndex, 1] = diff_color_val.y;
    diff_color[pixelIndex, 2] = diff_color_val.z;

    spec_color[pixelIndex, 0] = spec_color_val.x;
    spec_color[pixelIndex, 1] = spec_color_val.y;
    spec_color[pixelIndex, 2] = spec_color_val.z;
}