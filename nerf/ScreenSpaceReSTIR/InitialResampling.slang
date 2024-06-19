import utils.lightDi;
import utils.random;
import utils.helperDi;
import utils.res;
import utils.brdf;

static const uint kLightTileCount = LIGHT_TILE_COUNT;
static const uint kLightTileSize = LIGHT_TILE_SIZE;

static const uint kScreenTileSize = SCREEN_TILE_SIZE;
static const uint kInitialLightSampleCount = INITIAL_LIGHT_SAMPLE_COUNT;
static const uint kInitialBRDFSampleCount = INITIAL_BRDF_SAMPLE_COUNT;
static const float kBRDFSampleRatio = float(kInitialBRDFSampleCount) / float(kInitialLightSampleCount + kInitialBRDFSampleCount);
static const bool kSpecularOnly = false;
static const bool kCheckVisibility = true;

#define VIS_near 0.01f

[AutoPyBindCUDA]
[CUDAKernel]
void process_InitialResampling(Reservoir reservoirs, TensorView<float> env_tex,
    uint env_width, uint env_height, uint framedim_x, uint framedim_y, uint frameIndex,
    TensorView<float> occ_map, TensorView<float> pos_map, TensorView<float> normal_depth, 
    TensorView<float> brdf_map, TensorView<float> ray_dir, 
    float weightSum,
    TensorView<float> light_data, TensorView<int> light_uv, TensorView<float> light_inv_pdf
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        reservoirs.light_data[pixelIndex, 0] = 0.f;
        reservoirs.light_data[pixelIndex, 1] = 0.f;
        reservoirs.light_data[pixelIndex, 2] = 0.f;
        reservoirs.light_pdf[pixelIndex, 0] = 0.f;
        reservoirs.M[pixelIndex, 0] = 0;
        reservoirs.weight[pixelIndex, 0] = 0.f;
        return;
    }

    uint2 screenTile = pixel / kScreenTileSize;

    uint tileSg = Seed_Generator(screenTile, frameIndex);
    uint tileIndex = min(uint(sampleNext1D(tileSg) * kLightTileCount), kLightTileCount - 1);
    uint lightTileOffset = tileIndex * kLightTileSize;

    uint sg = Seed_Generator(pixel, frameIndex);
    uint stride = (kLightTileSize + kInitialLightSampleCount - 1) / kInitialLightSampleCount;
    uint offset = min(uint(sampleNext1D(sg) * stride), stride - 1);

    float3 curr_normal = float3(normal_depth[pixelIndex,0], normal_depth[pixelIndex,1], normal_depth[pixelIndex,2]);
    float curr_depth = normal_depth[pixelIndex,3];
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_brdf = float3(brdf_map[pixelIndex,0], brdf_map[pixelIndex,1], brdf_map[pixelIndex,2]);

    RisState risState = createEmpty_RisState();
    // Resample light samples from the light tile.
    for (uint i = 0; i < kInitialLightSampleCount; ++i)
    {
        // Pick light sample and evaluate it.
        uint index = lightTileOffset + (offset + i * stride) % kLightTileSize;
        PresampledLight lightSample = createInvalid_PresampledLight();
        lightSample.light_data.x = light_data[index,0];
        lightSample.light_data.y = light_data[index,1];
        lightSample.light_data.z = light_data[index,2];
        lightSample.light_uv.x = light_uv[index,0];
        lightSample.light_uv.y = light_uv[index,1];
        lightSample.inv_pdf = light_inv_pdf[index];

    
        // Resample light sample.
        float3 light_emission, light_dir;
        get_light_info(env_tex, env_width, env_height, float2(lightSample.light_data.y,lightSample.light_data.z), light_emission, light_dir);
        float targetPdf = evalTargetFunction(light_emission, light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
        float sourcePdf = evalInitialSamplePdf(kSpecularOnly, kInitialBRDFSampleCount, kBRDFSampleRatio, light_dir, lightSample.inv_pdf, 
            -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
        streamingResampleStep(risState, lightSample, targetPdf, sourcePdf, sg);
    }

        // Resample BRDF samples.
    if (kInitialBRDFSampleCount > 0)
    {
        for (uint i = 0; i < kInitialBRDFSampleCount; ++i)
        {
            PresampledLight lightSample = createInvalid_PresampledLight();
            float3 dir;
            if (sampleBRDF(kSpecularOnly, sampleNext3D(sg), dir, -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y))
            {
                float brdfPdf = evalPdfBRDF(kSpecularOnly, dir, -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
                float light_pdf = env_light_pdf(dir, env_tex, env_width, env_height, weightSum);
                lightSample.light_data.x = 1.0f;
                lightSample.inv_pdf = light_pdf;
                float2 brdf_light_uv = oct_encode(dir);
                lightSample.light_data.y = brdf_light_uv.x;
                lightSample.light_data.z = brdf_light_uv.y;
            }

            if(lightSample.light_data.x < 0.1f)
            {
                risState.M += 1.f;
                continue;
            }

            // Resample BRDF light sample.
            float3 light_emission = env_le(dir, env_tex, env_width, env_height);
            float targetPdf = evalTargetFunction(light_emission, dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
            float sourcePdf = evalInitialSamplePdf(kSpecularOnly, kInitialBRDFSampleCount, kBRDFSampleRatio, dir, lightSample.inv_pdf, 
            -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
            streamingResampleStep(risState, lightSample, targetPdf, sourcePdf, sg);
        }
    }

    // Check visibility.
    //if (kCheckVisibility && risState.light_data.x>0.1f)
    //{
        //const EvaluatedLightSample evaluatedLightSample = lights.evalLightSample(risState.lightSample, evalContext.pos);
        //if (evalContext.evalVisibility(evaluatedLightSample) == 0.f) risState = RisState::createEmpty();
    //}

    // Instead of storing the weight sum and weight individually,
    // we precompute the term that would be multiplied with the final sample evaluation.
    risState.weight = risState.weight > 0.f ? (risState.weightSum / risState.M) / risState.weight : 0.f;
    risState.M = 1.f;

    // Save reservoir.
    reservoirs.light_data[pixelIndex, 0] = risState.light_data.x;
    reservoirs.light_data[pixelIndex, 1] = risState.light_data.y;
    reservoirs.light_data[pixelIndex, 2] = risState.light_data.z;
    reservoirs.light_pdf[pixelIndex, 0] = risState.inv_pdf;
    reservoirs.M[pixelIndex, 0] = int(risState.M);
    reservoirs.weight[pixelIndex, 0] = risState.weight;

    if (isinf(risState.weight) || isnan(risState.weight))
    {
        reservoirs.light_data[pixelIndex, 0] = 0.f;
        reservoirs.light_data[pixelIndex, 1] = 0.f;
        reservoirs.light_data[pixelIndex, 2] = 0.f;
        reservoirs.M[pixelIndex, 0] = 0;
        reservoirs.weight[pixelIndex, 0] = 0.f;
    }

}

[AutoPyBindCUDA]
[CUDAKernel]
void process_InitialResampling_(
                                 TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                                 TensorView<float> vert, TensorView<int> v_indx, TensorView<float> pos_map,
                                  Reservoir reservoirs, TensorView<float> env_tex,
                                  uint env_width, uint env_height, uint framedim_x, uint framedim_y, uint frameIndex,
                                  TensorView<float> occ_map, TensorView<float> normal_depth,
                                  TensorView<float> brdf_map, TensorView<float> ray_dir,
                                  TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_,
                                  TensorView<float> light_data, TensorView<int> light_uv, TensorView<float> light_inv_pdf
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        reservoirs.light_data[pixelIndex, 0] = 0.f;
        reservoirs.light_data[pixelIndex, 1] = 0.f;
        reservoirs.light_data[pixelIndex, 2] = 0.f;
        reservoirs.light_pdf[pixelIndex, 0] = 0.f;
        reservoirs.M[pixelIndex, 0] = 0;
        reservoirs.weight[pixelIndex, 0] = 0.f;

        return;
    }

    uint2 screenTile = pixel / kScreenTileSize;

    uint tileSg = Seed_Generator(screenTile, frameIndex);
    uint tileIndex = min(uint(sampleNext1D(tileSg) * kLightTileCount), kLightTileCount - 1);
    uint lightTileOffset = tileIndex * kLightTileSize;

    uint sg = Seed_Generator(pixel, frameIndex);
    uint stride = (kLightTileSize + kInitialLightSampleCount - 1) / kInitialLightSampleCount;
    uint offset = min(uint(sampleNext1D(sg) * stride), stride - 1);

    float3 curr_normal = float3(normal_depth[pixelIndex,0], normal_depth[pixelIndex,1], normal_depth[pixelIndex,2]);
    float curr_depth = normal_depth[pixelIndex,3];
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_brdf = float3(brdf_map[pixelIndex,0], brdf_map[pixelIndex,1], brdf_map[pixelIndex,2]);

    RisState risState = createEmpty_RisState();
    // Resample light samples from the light tile.
    for (uint i = 0; i < kInitialLightSampleCount; ++i)
    {
        // Pick light sample and evaluate it.
        uint index = lightTileOffset + (offset + i * stride) % kLightTileSize;
        PresampledLight lightSample = createInvalid_PresampledLight();
        lightSample.light_data.x = light_data[index,0];
        lightSample.light_data.y = light_data[index,1];
        lightSample.light_data.z = light_data[index,2];
        lightSample.light_uv.x = light_uv[index,0];
        lightSample.light_uv.y = light_uv[index,1];
        lightSample.inv_pdf = light_inv_pdf[index];

    
        // Resample light sample.
        float3 light_emission, light_dir;
        get_light_info(env_tex, env_width, env_height, float2(lightSample.light_data.y,lightSample.light_data.z), light_emission, light_dir);
        float targetPdf = evalTargetFunction(light_emission, light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
        float sourcePdf = evalInitialSamplePdf(kSpecularOnly, kInitialBRDFSampleCount, kBRDFSampleRatio, light_dir, lightSample.inv_pdf, 
            -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
        streamingResampleStep(risState, lightSample, targetPdf, sourcePdf, sg);
    }

        // Resample BRDF samples.
    if (kInitialBRDFSampleCount > 0)
    {
        for (uint i = 0; i < kInitialBRDFSampleCount; ++i)
        {
            PresampledLight lightSample = createInvalid_PresampledLight();
            float3 dir;
            if (sampleBRDF(kSpecularOnly, sampleNext3D(sg), dir, -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y))
            {
                float brdfPdf = evalPdfBRDF(kSpecularOnly, dir, -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
                float light_pdf = sample_light_pdf(dir, env_width, env_height, pdf_, mpdf_);
                lightSample.light_data.x = 1.0f;
                lightSample.inv_pdf = light_pdf;
                float2 brdf_light_uv = oct_encode(dir);
                lightSample.light_data.y = brdf_light_uv.x;
                lightSample.light_data.z = brdf_light_uv.y;
            }

            if(lightSample.light_data.x < 0.1f)
            {
                risState.M += 1.f;
                continue;
            }

            // Resample BRDF light sample.
            //float3 light_emission = env_le(dir, env_tex, env_width, env_height);
            float3 temp_dir = dir;
            float3 r_dir = ngp_dir(temp_dir);
            float3 light_emission = env_le(r_dir, env_tex, env_width, env_height);
            float targetPdf = evalTargetFunction(light_emission, dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
            float sourcePdf = evalInitialSamplePdf(kSpecularOnly, kInitialBRDFSampleCount, kBRDFSampleRatio, dir, lightSample.inv_pdf, 
            -curr_ray_dir, curr_normal, curr_brdf.z, curr_brdf.x, curr_brdf.y);
            streamingResampleStep(risState, lightSample, targetPdf, sourcePdf, sg);
        }
    }

    // Check visibility.
    if (kCheckVisibility && risState.light_data.x>0.1f)
    {
        float3 shadow_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);
        float3 light_dir = oct_decode(float2(risState.light_data.y,risState.light_data.z));
        float t_min = 0.f;
        float t_max = 1e7;
        float t_hit = 0.f;
        float3 pos = float3(0.f);
        float vis_near = VIS_near;
        shadow_pos = shadow_pos + vis_near*light_dir;
        bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       shadow_pos, light_dir, t_min, t_max, t_hit, pos);

        if (hit) risState = createEmpty_RisState();
    }

    // Instead of storing the weight sum and weight individually,
    // we precompute the term that would be multiplied with the final sample evaluation.
    risState.weight = risState.weight > 0.f ? (risState.weightSum / risState.M) / risState.weight : 0.f;
    risState.M = 1.f;
    
    // Save reservoir.
    reservoirs.light_data[pixelIndex, 0] = risState.light_data.x;
    reservoirs.light_data[pixelIndex, 1] = risState.light_data.y;
    reservoirs.light_data[pixelIndex, 2] = risState.light_data.z;
    reservoirs.light_pdf[pixelIndex, 0] = risState.inv_pdf;
    reservoirs.M[pixelIndex, 0] = int(risState.M);
    reservoirs.weight[pixelIndex, 0] = risState.weight;

    if (isinf(risState.weight) || isnan(risState.weight))
    {
        reservoirs.light_data[pixelIndex, 0] = 0.f;
        reservoirs.light_data[pixelIndex, 1] = 0.f;
        reservoirs.light_data[pixelIndex, 2] = 0.f;
        reservoirs.light_pdf[pixelIndex, 0] = 0.f;
        reservoirs.M[pixelIndex, 0] = 0;
        reservoirs.weight[pixelIndex, 0] = 0.f;
    }

}