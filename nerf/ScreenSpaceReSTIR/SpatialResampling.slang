import utils.lightDi;
import utils.random;
import utils.helperDi;
import utils.res;
import utils.brdf;

static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
static const uint neighborCount = NEIGHBOR_COUNT;
static const float gatherRadius = GATHER_RADIUS;
static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;
static const bool kUsePairwiseMIS = bool(1);
static const bool kUnbiased = bool(1);
static const float normalThreshold = 0.5f;
static const float depthThreshold = 0.1f;

#define VIS_near 0.01f

[PreferRecompute]
uint getPixelIndex(uint2 pixel, int framedim_x) { return pixel.y * framedim_x + pixel.x; }

[PreferRecompute]
bool isValidPixel(int2 pixel, int framedim_x, int framedim_y)
{ 
    if (pixel.x >= framedim_x || pixel.x < 0) return false;
    if (pixel.y >= framedim_y || pixel.y < 0) return false;
    return true;
}

[PreferRecompute]
bool isValidPixel(int2 pixel, uint2 frameDim) { return all(pixel >= 0 && pixel < frameDim); }

[PreferRecompute]
int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i, TensorView<float> neighborOffsets)
{
    uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
    int2 neighborPixel = pixel + int2(float2(neighborOffsets[neighborIndex,0] * gatherRadius, neighborOffsets[neighborIndex,1] * gatherRadius));

    return neighborPixel;
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_SpatialResampling(Reservoir reservoirs, Reservoir prevReservoirs, TensorView<float> neighborOffsets,
    TensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y, uint frameIndex,
    TensorView<float> occ_map, TensorView<float> normal_depth, TensorView<float> brdf_map, TensorView<float> ray_dir,
    TensorView<float> vis_map, no_diff float thr
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
    uint sg = Seed_Generator(pixel, frameIndex);

    //currentContext
    float3 curr_normal = float3(normal_depth[pixelIndex,0], normal_depth[pixelIndex,1], normal_depth[pixelIndex,2]);
    float curr_depth = normal_depth[pixelIndex,3];
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_brdf = float3(brdf_map[pixelIndex,0], brdf_map[pixelIndex,1], brdf_map[pixelIndex,2]);

    // Setup resampling.
    RisState risState = createEmpty_RisState();
    const uint startIndex = uint(sampleNext1D(sg) * kNeighborOffsetCount);
    uint validCount = 0;

    uint sg2 = Seed_Generator(int2(0, 0), frameIndex + 1);
    if (kUsePairwiseMIS)
    {
        reservoir currentReservoir;
        currentReservoir.light_data = 
        float3(prevReservoirs.light_data[pixelIndex, 0], prevReservoirs.light_data[pixelIndex, 1], prevReservoirs.light_data[pixelIndex, 2]);
        currentReservoir.light_pdf = prevReservoirs.light_pdf[pixelIndex, 0];
        currentReservoir.M = prevReservoirs.M[pixelIndex, 0];
        currentReservoir.weight = prevReservoirs.weight[pixelIndex, 0];

        float3 curr_light_emission, curr_light_dir;
        get_light_info(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), curr_light_emission, curr_light_dir);
        float currentTargetPdf = evalTargetFunction(curr_light_emission, curr_light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);

        risState.canonicalWeight = 1.f;
        uint validNeighbors = 1;
        
        for (uint i = 0; i < neighborCount; ++i)
        {
            // Get neighbor pixel.
            int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i, neighborOffsets);
            if (!isValidPixel(neighborPixel, uint2(framedim_x, framedim_y))) continue;
            uint neighborPixelIndex = getPixelIndex(neighborPixel, framedim_x);

            // Check for valid neighbor.
            float3 neighborNormal = float3(normal_depth[neighborPixelIndex,0], normal_depth[neighborPixelIndex,1], 
                normal_depth[neighborPixelIndex,2]);
            float neighborDepth = normal_depth[neighborPixelIndex,3];
            if (!isValidNeighbor(curr_normal, curr_depth, neighborNormal, neighborDepth, normalThreshold, depthThreshold)) continue;

            reservoir neighborReservoir;
            neighborReservoir.light_data = 
            float3(prevReservoirs.light_data[neighborPixelIndex, 0], prevReservoirs.light_data[neighborPixelIndex, 1], prevReservoirs.light_data[neighborPixelIndex, 2]);
            neighborReservoir.light_pdf = prevReservoirs.light_pdf[neighborPixelIndex, 0];
            neighborReservoir.M = prevReservoirs.M[neighborPixelIndex, 0];
            neighborReservoir.weight = prevReservoirs.weight[neighborPixelIndex, 0];
            if (neighborReservoir.M == 0) continue;

            // Setup neighbor evalulation context and resample.
            float3 neighbor_ray_dir = float3(ray_dir[neighborPixelIndex,0], ray_dir[neighborPixelIndex,1], ray_dir[neighborPixelIndex,2]);
            float3 neighbor_brdf = float3(brdf_map[neighborPixelIndex,0], brdf_map[neighborPixelIndex,1], brdf_map[neighborPixelIndex,2]);
            if(occ_map[neighborPixelIndex,0]<0.1f) continue;

            ++validNeighbors;
            ++validCount;

            float3 neighbor_light_emission, neighbor_light_dir;
            get_light_info(env_tex, env_width, env_height, float2(neighborReservoir.light_data.y,neighborReservoir.light_data.z), neighbor_light_emission, neighbor_light_dir);

            if (kUnbiased)
                {
                    float candidateVisibility = 1.0f;
                    float canonicalVisibility = 1.0f;
                    /*
                    float candidateVisibility = vis_map[pixelIndex, i];
                    float canonicalVisibility = vis_map[pixelIndex, neighborCount + i];
                    if (candidateVisibility < thr) candidateVisibility = 0.f;
                    else candidateVisibility = 1.f;
                    if (canonicalVisibility < thr) canonicalVisibility = 0.f;
                    else canonicalVisibility = 1.f;
                    */
                    streamingResampleStepMisUnbiased(risState, currentReservoir, curr_light_emission, curr_light_dir,
                    currentTargetPdf, curr_normal, curr_ray_dir, curr_brdf, 
                    neighborReservoir, neighbor_light_emission, neighbor_light_dir, neighborNormal, neighbor_ray_dir, neighbor_brdf,
                    canonicalVisibility, candidateVisibility,
                    neighborCount, sg);
                }
            else
                streamingResampleStepMis(risState, currentReservoir, curr_light_emission, curr_light_dir,
                currentTargetPdf, curr_normal, curr_ray_dir, curr_brdf, 
                neighborReservoir, neighbor_light_emission, neighbor_light_dir, neighborNormal, neighbor_ray_dir, neighbor_brdf, 
                neighborCount, sg);
        }
        streamingResampleFinalizeMis(risState, currentReservoir, currentTargetPdf, sg);
        risState.M = currentReservoir.M;
        risState.weight = risState.weight > 0.f ? (risState.weightSum / validNeighbors) / risState.weight : 0.f;
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

[AutoPyBindCUDA]
[CUDAKernel]
void process_SpatialResampling_(
                                TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                                TensorView<float> vert, TensorView<int> v_indx, TensorView<float> pos_map,
                                Reservoir reservoirs, Reservoir prevReservoirs, TensorView<float> neighborOffsets,
                               TensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y, uint frameIndex,
                               TensorView<float> occ_map, TensorView<float> normal_depth, TensorView<float> brdf_map, TensorView<float> ray_dir
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
    uint sg = Seed_Generator(pixel, frameIndex);

    //currentContext
    float3 curr_normal = float3(normal_depth[pixelIndex,0], normal_depth[pixelIndex,1], normal_depth[pixelIndex,2]);
    float curr_depth = normal_depth[pixelIndex,3];
    float3 curr_ray_dir = float3(ray_dir[pixelIndex,0], ray_dir[pixelIndex,1], ray_dir[pixelIndex,2]);
    float3 curr_brdf = float3(brdf_map[pixelIndex,0], brdf_map[pixelIndex,1], brdf_map[pixelIndex,2]);

    // Setup resampling.
    RisState risState = createEmpty_RisState();
    const uint startIndex = uint(sampleNext1D(sg) * kNeighborOffsetCount);
    uint validCount = 0;

    uint sg2 = Seed_Generator(int2(0, 0), frameIndex + 1);
    if (kUsePairwiseMIS)
    {
        reservoir currentReservoir;
        currentReservoir.light_data = 
        float3(prevReservoirs.light_data[pixelIndex, 0], prevReservoirs.light_data[pixelIndex, 1], prevReservoirs.light_data[pixelIndex, 2]);
        currentReservoir.light_pdf = prevReservoirs.light_pdf[pixelIndex, 0];
        currentReservoir.M = prevReservoirs.M[pixelIndex, 0];
        currentReservoir.weight = prevReservoirs.weight[pixelIndex, 0];

        float3 curr_light_emission, curr_light_dir;
        get_light_info(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), curr_light_emission, curr_light_dir);
        float currentTargetPdf = evalTargetFunction(curr_light_emission, curr_light_dir, curr_normal, curr_ray_dir, curr_brdf, 1.0f);

        float3 curr_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);

        risState.canonicalWeight = 1.f;
        uint validNeighbors = 1;
        
        for (uint i = 0; i < neighborCount; ++i)
        {
            // Get neighbor pixel.
            int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i, neighborOffsets);
            if (!isValidPixel(neighborPixel, uint2(framedim_x, framedim_y))) continue;
            uint neighborPixelIndex = getPixelIndex(neighborPixel, framedim_x);

            // Check for valid neighbor.
            float3 neighborNormal = float3(normal_depth[neighborPixelIndex,0], normal_depth[neighborPixelIndex,1], 
                normal_depth[neighborPixelIndex,2]);
            float neighborDepth = normal_depth[neighborPixelIndex,3];
            if (!isValidNeighbor(curr_normal, curr_depth, neighborNormal, neighborDepth, normalThreshold, depthThreshold)) continue;

            reservoir neighborReservoir;
            neighborReservoir.light_data = 
            float3(prevReservoirs.light_data[neighborPixelIndex, 0], prevReservoirs.light_data[neighborPixelIndex, 1], prevReservoirs.light_data[neighborPixelIndex, 2]);
            neighborReservoir.light_pdf = prevReservoirs.light_pdf[neighborPixelIndex, 0];
            neighborReservoir.M = prevReservoirs.M[neighborPixelIndex, 0];
            neighborReservoir.weight = prevReservoirs.weight[neighborPixelIndex, 0];
            if (neighborReservoir.M == 0) continue;

            // Setup neighbor evalulation context and resample.
            float3 neighbor_ray_dir = float3(ray_dir[neighborPixelIndex,0], ray_dir[neighborPixelIndex,1], ray_dir[neighborPixelIndex,2]);
            float3 neighbor_brdf = float3(brdf_map[neighborPixelIndex,0], brdf_map[neighborPixelIndex,1], brdf_map[neighborPixelIndex,2]);
            if(occ_map[neighborPixelIndex,0]<0.1f) continue;

            ++validNeighbors;
            ++validCount;

            float3 neighbor_light_emission, neighbor_light_dir;
            get_light_info(env_tex, env_width, env_height, float2(neighborReservoir.light_data.y,neighborReservoir.light_data.z), neighbor_light_emission, neighbor_light_dir);

            if (kUnbiased)
            {
                float3 neighbor_pos = float3(pos_map[neighborPixelIndex, 0], pos_map[neighborPixelIndex, 1], pos_map[neighborPixelIndex, 2]);
                float t_min = 0.f;
                float t_max = 1e7;
                float t_hit = 0.f;
                float3 pos_test = float3(0.f);

                float vis_near = VIS_near;
                float3 now_pos = curr_pos + vis_near*neighbor_light_dir;
                neighbor_pos = neighbor_pos + vis_near*curr_light_dir;

                bool canonical_hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       now_pos, neighbor_light_dir, t_min, t_max, t_hit, pos_test);
                bool candidate_hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       neighbor_pos, curr_light_dir, t_min, t_max, t_hit, pos_test);

                float candidateVisibility = 1.0f;
                float canonicalVisibility = 1.0f;
                if(canonical_hit) canonicalVisibility = 0.f;
                if(candidate_hit) candidateVisibility = 0.f;
                streamingResampleStepMisUnbiased(risState, currentReservoir, curr_light_emission, curr_light_dir,
                currentTargetPdf, curr_normal, curr_ray_dir, curr_brdf, 
                neighborReservoir, neighbor_light_emission, neighbor_light_dir, neighborNormal, neighbor_ray_dir, neighbor_brdf,
                canonicalVisibility, candidateVisibility,
                neighborCount, sg);
            }
            else
                streamingResampleStepMis(risState, currentReservoir, curr_light_emission, curr_light_dir,
                currentTargetPdf, curr_normal, curr_ray_dir, curr_brdf, 
                neighborReservoir, neighbor_light_emission, neighbor_light_dir, neighborNormal, neighbor_ray_dir, neighbor_brdf, 
                neighborCount, sg);
        }
        streamingResampleFinalizeMis(risState, currentReservoir, currentTargetPdf, sg);
        risState.M = currentReservoir.M;
        risState.weight = risState.weight > 0.f ? (risState.weightSum / validNeighbors) / risState.weight : 0.f;
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