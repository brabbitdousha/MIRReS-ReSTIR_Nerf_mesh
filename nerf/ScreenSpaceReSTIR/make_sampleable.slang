import utils.helper;

[PreferRecompute]
float3 scene_env_tex_sample(TensorView<float> env_tex, float2 uv, uint width, uint height)
{
    return eval_bi(env_tex,uv,width,height);
}

[PreferRecompute]
float3 env_le(float3 dir, TensorView<float> env_tex, uint width, uint height){
    const float TWO_PI = 6.2831853f;
    const float INV_TWO_PI = 0.1591549f;
    const float INV_PI = 0.31830988f;
    float theta = acos(dir.y);
    float sin_theta = sin(theta);
    if(abs(sin_theta) < 1e-4)
        return float3(0,0,0);
    float phi = atan2(dir.z, dir.x);
    if(phi < 0)
        phi += TWO_PI;
    float2 uv = float2(phi * INV_TWO_PI, 1 - theta * INV_PI);
    return scene_env_tex_sample(env_tex, uv, width, height);
}

[PreferRecompute]
float3 ngp_dir(float3 curr_ray_dir)
{
    float3 new_dir = float3(-curr_ray_dir.x, curr_ray_dir.z, curr_ray_dir.y);
    return new_dir;
}

[AutoPyBindCUDA]
[CUDAKernel]
void make_sampleable(TensorView<float> env_tex, TensorView<float> weight, uint width, uint height)
{
    // Get the 'global' index of this thread.
    const float PI = 3.141592653589793f;
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint h = dispatchIdx.x;
    uint w = dispatchIdx.y;
    if (h >= height || w >= width) return;

    float v = (h + .5f) / height;
    float sin_theta = sin(PI * v);
    float2 uv = float2((w + .5f) / width, v);
    float theta = uv.y * PI, phi = uv.x * 2 * PI;
    float cos_theta = cos(theta), cos_phi = cos(phi), sin_theta_dir = sin(theta),
          sin_phi = sin(phi);
    
    float3 raw_dir = float3(sin_theta_dir * cos_phi, cos_theta, sin_theta_dir * sin_phi);
    float3 dir = ngp_dir(raw_dir);


    //weight[h * width + w] = luminance(eval_bi(env_tex, float2((w + .5f) / width, 1.f - v), width, height));
    weight[h * width + w] = luminance(env_le(dir, env_tex, width, height));
    weight[h * width + w] *= sin_theta;

}

[AutoPyBindCUDA]
[CUDAKernel]
void Distribution2D(uint w, uint h, TensorView<float> pdf_, TensorView<float> cdf_)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint y = dispatchIdx.x;
    uint x = dispatchIdx.y;

    if (y >= h || x>=w ) return;
    uint ip = y * w;
    uint ic = y * (w + 1);
    uint tail = ic + w;
    float row_weight = cdf_[tail];
    ip+=x;
    ic+=x;
    if(row_weight < 1e-4f) 
    {
        pdf_[ip] = 1.0f / w;
        cdf_[ic] = x / float(w);

    } else 
    {
        pdf_[ip] /= row_weight;
        cdf_[ic] /= row_weight;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void build_alias_map(TensorView<int> partition_table, TensorView<float> q,TensorView<int> alias,
    uint size)
{
    int s = 0;
    int large = size;
    for(int i = 0; i < size; ++i) {
        partition_table[(q[i] < 1.0f) ? (s++) : (--large)] = i;
    }
    for(s = 0; s < large && large < size; ++s) 
    {
        const int j = partition_table[s];
        const int k = partition_table[large];
        alias[j] = k;
        q[k] += q[j] - 1.0f;
        large = (q[k] < 1.0f) ? (large + 1) : large;
    }

}

[AutoPyBindCUDA]
[CUDAKernel]
void build_alias_map_2(TensorView<int> alias, TensorView<float> q,TensorView<float> pdf, TensorView<float> alias_map,
    uint size, uint width, uint height)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint i = dispatchIdx.x;
    if(i>=size) return;
    int h = i/width;
    int w = i%width;
    h = height - h - 1;
    int tidx = h * width + w;
    alias_map[i , 0] = q[tidx];
    alias_map[i , 1] = float(alias[tidx]);
    alias_map[i , 2] = pdf[tidx];
    alias_map[i , 3] = pdf[alias[tidx]];

}

[AutoPyBindCUDA]
[CUDAKernel]
void buildEnvLightAliasTable(TensorView<float> env_tex, TensorView<float> weight, uint width, uint height)
{
    // Get the 'global' index of this thread.
    const float M_PI = 3.141592653589793f;
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint y = dispatchIdx.x;
    uint x = dispatchIdx.y;
    if (y >= height || x >= width) return;

    float theta = (float)M_PI * (y + 0.5f) / height;
    float solidAngle = (2.f * (float)M_PI / width) * ((float)M_PI / height) * sin(theta);
    weight[y * width + x] = solidAngle * luminance(eval_bi(env_tex,float2((x + .5f) / width, (y + .5f) / height), width, height));
}

[AutoPyBindCUDA]
[CUDAKernel]
void AliasTable(TensorView<float> items, TensorView<float> lowIdx, TensorView<float> highIdx, TensorView<float> weights, 
    float avgWeight, uint mCount, uint lowCount, uint highCount)
{
    for (int i = 0; i < mCount; ++i)
    {
        // Usual case:  We have an above-average and below-average sample we can combine into one alias table entry
        if ((lowIdx[i] > -1.f) && (highIdx[i] > -1.f))
        {
            // Create an alias table tuple: 
            items[i,0] = weights[int(lowIdx[i])] / avgWeight;
            items[i,1] = highIdx[i];
            items[i,2] = lowIdx[i];

            // We've removed some weight from element highIdx[i]; update it's weight, then re-enter it
            // on the end of either the above-average or below-average lists.
            float updatedWeight = (weights[int(lowIdx[i])] + weights[int(highIdx[i])]) - avgWeight;
            weights[int(highIdx[i])] = updatedWeight;
            if (updatedWeight < avgWeight)
                lowIdx[lowCount++] = highIdx[i];
            else
                highIdx[highCount++] = highIdx[i];
        }
        else if (highIdx[i] > -1.f)
        {
            items[i,0] = 1.0f;
            items[i,1] = highIdx[i];
            items[i,2] = highIdx[i];
        }
        else if (lowIdx[i] > -1.f)
        {
            items[i,0] = 1.0f;
            items[i,1] = lowIdx[i];
            items[i,2] = lowIdx[i];
        }
    }
}


[AutoPyBindCUDA]
[CUDAKernel]
void createNeighborOffsetTexture(int sampleCount, TensorView<float> neighborOffsets)
{
    const int R = 254;
    const float phi2 = 1.f / 1.3247179572447f;
    float u = 0.5f;
    float v = 0.5f;
    for (uint32_t index = 0; index < sampleCount * 2;)
    {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.f) u -= 1.f;
        if (v >= 1.f) v -= 1.f;

        float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
        if (rSq > 0.25f) continue;

        neighborOffsets[index++] = float(int((u - 0.5f) * R));
        neighborOffsets[index++] = float(int((v - 0.5f) * R));
    }
}