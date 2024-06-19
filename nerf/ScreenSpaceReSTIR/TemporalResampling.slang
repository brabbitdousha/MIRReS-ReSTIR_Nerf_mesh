import utils.lightDi;
import utils.random;
import utils.helperDi;
import utils.res;
import utils.brdf;

static const uint kMaxHistoryLength = MAX_HISTORY_LENGTH;
static const bool kUsePairwiseMIS = bool(0);
static const bool kUnbiased = bool(0);
static const float normalThreshold = 0.5f;
static const float depthThreshold = 0.1f;

[PreferRecompute]
bool isValidPixel(int2 pixel, int framedim_x, int framedim_y)
{ 
    if (pixel.x >= framedim_x || pixel.x < 0) return false;
    if (pixel.y >= framedim_y || pixel.y < 0) return false;
    return true;
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_TemporalResampling(Reservoir reservoirs, Reservoir prevReservoirs,
                                TensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y, uint frameIndex,
                                TensorView<float> occ_map, TensorView<float> normal_depth, TensorView<float> brdf_map, TensorView<float> ray_dir,
                                TensorView<float> prev_occ_map, TensorView<float> prev_normal_depth, TensorView<float> prev_brdf_map, TensorView<float> prev_ray_dir,
                                TensorView<float> motionVectors
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) return;
    uint sg = Seed_Generator(pixel, frameIndex);
    
    // Back-project this pixel to previous frame.
    const int2 prevPixel = int2(pixel + float2(motionVectors[pixelIndex,0] * framedim_x, motionVectors[pixelIndex,1] * framedim_y) + (sampleNext2D(sg) * 1.f - 0.f));
    if (!isValidPixel(prevPixel, framedim_x, framedim_y)) return;
    const uint prevPixelIndex = prevPixel.y * framedim_x + prevPixel.x;
    if(prev_occ_map[prevPixelIndex,0]<0.1f) return;

    //currentContext
    float3 curr_normal = float3(normal_depth[pixelIndex,0], normal_depth[pixelIndex,1], normal_depth[pixelIndex,2]);
    float curr_depth = normal_depth[pixelIndex,3];
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_brdf = float3(brdf_map[pixelIndex,0], brdf_map[pixelIndex,1], brdf_map[pixelIndex,2]);
    //prevContext
    float3 pre_normal = float3(prev_normal_depth[prevPixelIndex,0], prev_normal_depth[prevPixelIndex,1], prev_normal_depth[prevPixelIndex,2]);
    float pre_depth = prev_normal_depth[prevPixelIndex,3];
    float3 pre_ray_dir = float3(prev_ray_dir[prevPixelIndex,0], prev_ray_dir[prevPixelIndex,1], prev_ray_dir[prevPixelIndex,2]);
    float3 pre_brdf = float3(prev_brdf_map[prevPixelIndex,0], prev_brdf_map[prevPixelIndex,1], prev_brdf_map[prevPixelIndex,2]);

    // Load reservoirs.
    reservoir currentReservoir;
    currentReservoir.light_data = 
    float3(reservoirs.light_data[pixelIndex, 0], reservoirs.light_data[pixelIndex, 1], reservoirs.light_data[pixelIndex, 2]);
    currentReservoir.light_pdf = reservoirs.light_pdf[pixelIndex, 0];
    currentReservoir.M = reservoirs.M[pixelIndex, 0];
    currentReservoir.weight = reservoirs.weight[pixelIndex, 0];

    reservoir prevReservoir;
    prevReservoir.light_data = 
    float3(prevReservoirs.light_data[prevPixelIndex, 0], prevReservoirs.light_data[prevPixelIndex, 1], prevReservoirs.light_data[prevPixelIndex, 2]);
    prevReservoir.light_pdf = prevReservoirs.light_pdf[prevPixelIndex, 0];
    prevReservoir.M = prevReservoirs.M[prevPixelIndex, 0];
    prevReservoir.weight = prevReservoirs.weight[prevPixelIndex, 0];
    prevReservoir.M = min(prevReservoir.M, currentReservoir.M * kMaxHistoryLength);

    // Setup resampling.
    RisState risState = createEmpty_RisState();

    if (kUsePairwiseMIS)
    {
        float3 light_emission, light_dir;
        get_light_info(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), light_emission, light_dir);

        float3 pre_light_emission, pre_light_dir;
        get_light_info(env_tex, env_width, env_height, float2(prevReservoir.light_data.y,prevReservoir.light_data.z), pre_light_emission, pre_light_dir);

        float currentTargetPdf = evalTargetFunction(light_emission, light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
        streamingResampleStepMis(risState, currentReservoir, light_emission, light_dir,
        currentTargetPdf, curr_normal, curr_ray_dir, curr_brdf, 
        prevReservoir, pre_light_emission, pre_light_dir, pre_normal, pre_ray_dir, pre_brdf, 
        1, sg);
        streamingResampleFinalizeMis(risState, currentReservoir, currentTargetPdf, sg);
        risState.weight = risState.weight > 0.f ? risState.weightSum / risState.weight : 0.f;
    }
    else
    {
        // Check if previous sample is valid for reuse.
        if (!kUnbiased && !isValidNeighbor(curr_normal, curr_depth, pre_normal, pre_depth, normalThreshold, depthThreshold)) return;
        
        // Resample current and previous reservoirs.
        float3 light_emission, light_dir;
        get_light_info(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), light_emission, light_dir);
        float targetPdf = evalTargetFunction(light_emission, light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
        streamingResampleStep(risState, currentReservoir, targetPdf, sg, true);

        float3 pre_light_emission, pre_light_dir;
        get_light_info(env_tex, env_width, env_height, float2(prevReservoir.light_data.y,prevReservoir.light_data.z), pre_light_emission, pre_light_dir);
        float pre_targetPdf = evalTargetFunction(pre_light_emission, pre_light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);
        bool usedPrev = streamingResampleStep(risState, prevReservoir, pre_targetPdf, sg, true);
        // Compute the unbiased normalization term (instead of using 1/M).
        float3 risState_light_emission, risState_light_dir;
        get_light_info(env_tex, env_width, env_height, float2(risState.light_data.y,risState.light_data.z), risState_light_emission, risState_light_dir);
        float currentPdf = evalTargetFunction(risState_light_emission, risState_light_dir, 
        curr_normal, curr_ray_dir, curr_brdf, 1.0f, kUnbiased);
        float prevPdf = evalTargetFunction(risState_light_emission, risState_light_dir, 
        pre_normal, pre_ray_dir, pre_brdf, 1.0f, kUnbiased);
        float normalization = (usedPrev ? prevPdf : currentPdf) / (currentReservoir.M * currentPdf + prevReservoir.M * prevPdf);
        risState.weight = risState.weight > 0.f ? (risState.weightSum * normalization) / risState.weight : 0.f;
    }

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