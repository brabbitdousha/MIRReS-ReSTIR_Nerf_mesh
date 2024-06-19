import utils.lightDi;
import utils.random;
import utils.helperDi;
import utils.res;
import utils.brdf;

static const bool kUseVisibility = bool(1);
static const bool kReuseVisibility = bool(0);

#define VIS_near 0.01f

[AutoPyBindCUDA]
[CUDAKernel]
void process_EvaluateFinalSamples(Reservoir reservoirs,
    TensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y,
    FinalSample finalSample, TensorView<float> vis_map, no_diff float thr
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;

    // Load reservoir.
    reservoir currentReservoir;
    currentReservoir.light_data = 
    float3(reservoirs.light_data[pixelIndex, 0], reservoirs.light_data[pixelIndex, 1], reservoirs.light_data[pixelIndex, 2]);
    currentReservoir.light_pdf = reservoirs.light_pdf[pixelIndex, 0];
    currentReservoir.M = reservoirs.M[pixelIndex, 0];
    currentReservoir.weight = reservoirs.weight[pixelIndex, 0];

    finalSample.dir[pixelIndex, 0] = 0.f;
    finalSample.dir[pixelIndex, 1] = 0.f;
    finalSample.dir[pixelIndex, 2] = 0.f;
    finalSample.distance[pixelIndex, 0] = 0.f;
    finalSample.Li[pixelIndex, 0] = 0.f;
    finalSample.Li[pixelIndex, 1] = 0.f;
    finalSample.Li[pixelIndex, 2] = 0.f;

    if(currentReservoir.light_data.x > 0.1f)
    {
        float3 light_emission, light_dir;
        get_light_info(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), light_emission, light_dir);
        // Evaluate visibility.
        float visibility = 1.f;
        if(kUseVisibility) 
        {
            visibility = vis_map[pixelIndex, 0];
            //if (visibility < thr) visibility = 0.f;
            //else visibility = 1.f;
        }
        if (visibility > 0.f)
        {
            finalSample.dir[pixelIndex, 0] = light_dir.x;
            finalSample.dir[pixelIndex, 1] = light_dir.y;
            finalSample.dir[pixelIndex, 2] = light_dir.z;
            finalSample.distance[pixelIndex, 0] = 1e6;//max for envmap
            float3 Li = currentReservoir.weight * light_emission * visibility;
            finalSample.Li[pixelIndex, 0] = Li.x;
            finalSample.Li[pixelIndex, 1] = Li.y;
            finalSample.Li[pixelIndex, 2] = Li.z;
        }
        else
        {
            if (kReuseVisibility)
            {
                reservoirs.light_data[pixelIndex, 0] = 0.f;
                reservoirs.light_data[pixelIndex, 1] = 0.f;
                reservoirs.light_data[pixelIndex, 2] = 0.f;
                reservoirs.light_pdf[pixelIndex, 0] = 0.f;
                reservoirs.M[pixelIndex, 0] = 0;
                reservoirs.weight[pixelIndex, 0] = 0.f;
            }
        }

    }

}

[AutoPyBindCUDA]
[CUDAKernel]
void process_EvaluateFinalSamples_get_vis(
    TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
    TensorView<float> vert, TensorView<int> v_indx,
    Reservoir reservoirs,
    uint framedim_x, uint framedim_y,
    TensorView<float> pos_map, TensorView<float> vis_map
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;

    // Load reservoir.
    
    float3 currentReservoir_light_data = 
    float3(reservoirs.light_data[pixelIndex, 0], reservoirs.light_data[pixelIndex, 1], reservoirs.light_data[pixelIndex, 2]);

    vis_map[pixelIndex, 0] = 1.0f;

    if(currentReservoir_light_data.x > 0.1f)
    {
        // Evaluate visibility.
        float3 shadow_pos = float3(pos_map[pixelIndex, 0], pos_map[pixelIndex, 1], pos_map[pixelIndex, 2]);
        float3 light_dir = oct_decode(float2(currentReservoir_light_data.y,currentReservoir_light_data.z));
        float t_min = 0.f;
        float t_max = 1e7;
        float t_hit = 0.f;
        float vis_near = VIS_near;
        float3 pos = float3(0.f);
        shadow_pos = shadow_pos + vis_near*light_dir;
        bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       shadow_pos, light_dir, t_min, t_max, t_hit, pos);

        if (hit) vis_map[pixelIndex, 0] = 0.0f;
        else vis_map[pixelIndex, 0] = 1.0f;
    }

}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void process_EvaluateFinalSamples_di_(Reservoir reservoirs,
    DiffTensorView<float> env_tex, uint env_width, uint env_height, uint framedim_x, uint framedim_y,
    FinalSample finalSample, TensorView<float> vis_map
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;

    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;

    // Load reservoir.
    reservoir currentReservoir;
    currentReservoir.light_data = 
    float3(reservoirs.light_data[pixelIndex, 0], reservoirs.light_data[pixelIndex, 1], reservoirs.light_data[pixelIndex, 2]);
    currentReservoir.light_pdf = reservoirs.light_pdf[pixelIndex, 0];
    currentReservoir.M = reservoirs.M[pixelIndex, 0];
    currentReservoir.weight = reservoirs.weight[pixelIndex, 0];

    finalSample.dir[pixelIndex, 0] = 0.f;
    finalSample.dir[pixelIndex, 1] = 0.f;
    finalSample.dir[pixelIndex, 2] = 0.f;
    finalSample.distance[pixelIndex, 0] = 0.f;
    float3 Li = float3(0.f);

    if(currentReservoir.light_data.x > 0.1f)
    {
        float3 light_emission, light_dir;
        get_light_info_di(env_tex, env_width, env_height, float2(currentReservoir.light_data.y,currentReservoir.light_data.z), light_emission, light_dir);
        // Evaluate visibility.
        float visibility = 1.f;
        if(kUseVisibility) visibility = vis_map[pixelIndex,0];// TODO
        if (visibility > 0.f)
        {
            finalSample.dir[pixelIndex, 0] = light_dir.x;
            finalSample.dir[pixelIndex, 1] = light_dir.y;
            finalSample.dir[pixelIndex, 2] = light_dir.z;
            finalSample.distance[pixelIndex, 0] = 1e6;//max for envmap
            Li = currentReservoir.weight * light_emission;
        }
        else
        {
            if (kReuseVisibility)
            {
                reservoirs.light_data[pixelIndex, 0] = 0.f;
                reservoirs.light_data[pixelIndex, 1] = 0.f;
                reservoirs.light_data[pixelIndex, 2] = 0.f;
                reservoirs.light_pdf[pixelIndex, 0] = 0.f;
                reservoirs.M[pixelIndex, 0] = 0;
                reservoirs.weight[pixelIndex, 0] = 0.f;
            }
        }

    }
    finalSample.Li[pixelIndex, 0] = Li.x;
    finalSample.Li[pixelIndex, 1] = Li.y;
    finalSample.Li[pixelIndex, 2] = Li.z;

}