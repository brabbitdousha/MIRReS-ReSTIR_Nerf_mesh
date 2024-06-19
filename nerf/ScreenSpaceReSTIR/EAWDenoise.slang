struct phi
{
    float c_phi, n_phi, p_phi;
};

[PreferRecompute]
bool isValidPixel(int2 pixel, uint2 frameDim) { return all(pixel >= 0 && pixel < frameDim); }

[Differentiable]
float getInputElement(DiffTensorView<float> input,
                      no_diff int2 uv)
{
    return input.load(uv);
}

[PreferRecompute]
float getInputElement_nf(TensorView<float> input,
                      int2 uv)
{
    return input.load(uv);
}

[PreferRecompute]
float3 getInputElement_3_nf(TensorView<float> input,
                      int framedim_x,
                      int2 uv)
{   
    uint uvIndex = uv.y * framedim_x + uv.x;
    float value_1 = getInputElement_nf(input, int2(uvIndex, 0));
    float value_2 = getInputElement_nf(input, int2(uvIndex, 1));
    float value_3 = getInputElement_nf(input, int2(uvIndex, 2));
    return float3(value_1,value_2,value_3);
}

[Differentiable]
float3 getInputElement_3(DiffTensorView<float> input,
                      no_diff int framedim_x,
                      no_diff int2 uv)
{   
    uint uvIndex = uv.y * framedim_x + uv.x;
    float value_1 = getInputElement(input, int2(uvIndex, 0));
    float value_2 = getInputElement(input, int2(uvIndex, 1));
    float value_3 = getInputElement(input, int2(uvIndex, 2));
    return float3(value_1,value_2,value_3);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void process_EAWDenoise(
    no_diff phi PHI,
    uint framedim_x, uint framedim_y, int stepWidth,
    TensorView<float> occ_map,
    DiffTensorView<float> color, DiffTensorView<float> normal_map, DiffTensorView<float> pos_map,
    DiffTensorView<float> out_color
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;
    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        out_color[pixelIndex,0] = no_diff color[pixelIndex,0];
        out_color[pixelIndex,1] = no_diff color[pixelIndex,1];
        out_color[pixelIndex,2] = no_diff color[pixelIndex,2];
        return;
    }
    float3 nval = float3(normal_map[pixelIndex,0], normal_map[pixelIndex,1], normal_map[pixelIndex,2]);
    float3 pval = float3(pos_map[pixelIndex,0], pos_map[pixelIndex,1], pos_map[pixelIndex,2]);
    float3 cval = float3(color[pixelIndex,0], color[pixelIndex,1], color[pixelIndex,2]);

    float c_phi = PHI.c_phi;//0.49f;
    float n_phi = PHI.n_phi;//0.1;
    float p_phi = PHI.p_phi;//0.00003;

    float2 offset[25];
    offset[0] = float2(-2,-2);
    offset[1] = float2(-1,-2);
    offset[2] = float2(0,-2);
    offset[3] = float2(1,-2);
    offset[4] = float2(2,-2);
    
    offset[5] = float2(-2,-1);
    offset[6] = float2(-1,-1);
    offset[7] = float2(0,-1);
    offset[8] = float2(1,-1);
    offset[9] = float2(2,-1);
    
    offset[10] = float2(-2,0);
    offset[11] = float2(-1,0);
    offset[12] = float2(0,0);
    offset[13] = float2(1,0);
    offset[14] = float2(2,0);
    
    offset[15] = float2(-2,1);
    offset[16] = float2(-1,1);
    offset[17] = float2(0,1);
    offset[18] = float2(1,1);
    offset[19] = float2(2,1);
    
    offset[20] = float2(-2,2);
    offset[21] = float2(-1,2);
    offset[22] = float2(0,2);
    offset[23] = float2(1,2);
    offset[24] = float2(2,2);

    float kernel[25];
    kernel[0] = 1.0f/256.0f;
    kernel[1] = 1.0f/64.0f;
    kernel[2] = 3.0f/128.0f;
    kernel[3] = 1.0f/64.0f;
    kernel[4] = 1.0f/256.0f;
    
    kernel[5] = 1.0f/64.0f;
    kernel[6] = 1.0f/16.0f;
    kernel[7] = 3.0f/32.0f;
    kernel[8] = 1.0f/16.0f;
    kernel[9] = 1.0f/64.0f;
    
    kernel[10] = 3.0f/128.0f;
    kernel[11] = 3.0f/32.0f;
    kernel[12] = 9.0f/64.0f;
    kernel[13] = 3.0f/32.0f;
    kernel[14] = 3.0f/128.0f;
    
    kernel[15] = 1.0f/64.0f;
    kernel[16] = 1.0f/16.0f;
    kernel[17] = 3.0f/32.0f;
    kernel[18] = 1.0f/16.0f;
    kernel[19] = 1.0f/64.0f;
    
    kernel[20] = 1.0f/256.0f;
    kernel[21] = 1.0f/64.0f;
    kernel[22] = 3.0f/128.0f;
    kernel[23] = 1.0f/64.0f;
    kernel[24] = 1.0f/256.0f;
    
    float3 sum = float3(0.f);
    float cum_w = 0.0;

    [ForceUnroll]
    for(int i = 0; i < 25; i++)
    {
        uint2 uv = pixel + int2(offset[i]*stepWidth);
        if(!isValidPixel(uv, uint2(framedim_x,framedim_y))) continue;
        
        float3 ctmp = getInputElement_3(color, framedim_x, uv);
        float3 t = cval - ctmp;
        float dist2 = dot(t,t);
        float c_w = min(exp(-(dist2)/c_phi), 1.0);

        float3 ntmp = getInputElement_3(normal_map, framedim_x, uv);
        t = nval - ntmp;
        dist2 = max(dot(t,t),0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);

        float3 ptmp = getInputElement_3(pos_map, framedim_x, uv);
        t = pval - ptmp;
        dist2 = max(dot(t,t), 0.0);
        float p_w = min(exp(-(dist2)/p_phi), 1.0);

        float weight = c_w*n_w*p_w;
        sum += ctmp*weight*kernel[i];
        cum_w += weight*kernel[i];
    }

    float3 sum_val = sum/cum_w;
    out_color[pixelIndex,0] = sum_val.x;
    out_color[pixelIndex,1] = sum_val.y;
    out_color[pixelIndex,2] = sum_val.z;

}

[AutoPyBindCUDA]
[CUDAKernel]
void process_EAWDenoise_no_di(
    phi PHI,
    uint framedim_x, uint framedim_y, int stepWidth,
    TensorView<float> occ_map,
    TensorView<float> color, TensorView<float> normal_map, TensorView<float> pos_map,
    TensorView<float> out_color
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;
    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        out_color[pixelIndex,0] = color[pixelIndex,0];
        out_color[pixelIndex,1] = color[pixelIndex,1];
        out_color[pixelIndex,2] = color[pixelIndex,2];
        return;
    }
    float3 nval = float3(normal_map[pixelIndex,0], normal_map[pixelIndex,1], normal_map[pixelIndex,2]);
    float3 pval = float3(pos_map[pixelIndex,0], pos_map[pixelIndex,1], pos_map[pixelIndex,2]);
    float3 cval = float3(color[pixelIndex,0], color[pixelIndex,1], color[pixelIndex,2]);

    float c_phi = PHI.c_phi;//0.49f;
    float n_phi = PHI.n_phi;//0.1;
    float p_phi = PHI.p_phi;//0.00003;

    float2 offset[25];
    offset[0] = float2(-2,-2);
    offset[1] = float2(-1,-2);
    offset[2] = float2(0,-2);
    offset[3] = float2(1,-2);
    offset[4] = float2(2,-2);
    
    offset[5] = float2(-2,-1);
    offset[6] = float2(-1,-1);
    offset[7] = float2(0,-1);
    offset[8] = float2(1,-1);
    offset[9] = float2(2,-1);
    
    offset[10] = float2(-2,0);
    offset[11] = float2(-1,0);
    offset[12] = float2(0,0);
    offset[13] = float2(1,0);
    offset[14] = float2(2,0);
    
    offset[15] = float2(-2,1);
    offset[16] = float2(-1,1);
    offset[17] = float2(0,1);
    offset[18] = float2(1,1);
    offset[19] = float2(2,1);
    
    offset[20] = float2(-2,2);
    offset[21] = float2(-1,2);
    offset[22] = float2(0,2);
    offset[23] = float2(1,2);
    offset[24] = float2(2,2);

    float kernel[25];
    kernel[0] = 1.0f/256.0f;
    kernel[1] = 1.0f/64.0f;
    kernel[2] = 3.0f/128.0f;
    kernel[3] = 1.0f/64.0f;
    kernel[4] = 1.0f/256.0f;
    
    kernel[5] = 1.0f/64.0f;
    kernel[6] = 1.0f/16.0f;
    kernel[7] = 3.0f/32.0f;
    kernel[8] = 1.0f/16.0f;
    kernel[9] = 1.0f/64.0f;
    
    kernel[10] = 3.0f/128.0f;
    kernel[11] = 3.0f/32.0f;
    kernel[12] = 9.0f/64.0f;
    kernel[13] = 3.0f/32.0f;
    kernel[14] = 3.0f/128.0f;
    
    kernel[15] = 1.0f/64.0f;
    kernel[16] = 1.0f/16.0f;
    kernel[17] = 3.0f/32.0f;
    kernel[18] = 1.0f/16.0f;
    kernel[19] = 1.0f/64.0f;
    
    kernel[20] = 1.0f/256.0f;
    kernel[21] = 1.0f/64.0f;
    kernel[22] = 3.0f/128.0f;
    kernel[23] = 1.0f/64.0f;
    kernel[24] = 1.0f/256.0f;
    
    float3 sum = float3(0.f);
    float cum_w = 0.0;

    [ForceUnroll]
    for(int i = 0; i < 25; i++)
    {
        uint2 uv = pixel + int2(offset[i]*stepWidth);
        if(!isValidPixel(uv, uint2(framedim_x,framedim_y))) continue;
        
        float3 ctmp = getInputElement_3_nf(color, framedim_x, uv);
        float3 t = cval - ctmp;
        float dist2 = dot(t,t);
        float c_w = min(exp(-(dist2)/c_phi), 1.0);

        float3 ntmp = getInputElement_3_nf(normal_map, framedim_x, uv);
        t = nval - ntmp;
        dist2 = max(dot(t,t),0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);

        float3 ptmp = getInputElement_3_nf(pos_map, framedim_x, uv);
        t = pval - ptmp;
        dist2 = max(dot(t,t), 0.0);
        float p_w = min(exp(-(dist2)/p_phi), 1.0);

        float weight = c_w*n_w*p_w;
        sum += ctmp*weight*kernel[i];
        cum_w += weight*kernel[i];
    }

    float3 sum_val = sum/cum_w;
    out_color[pixelIndex,0] = sum_val.x;
    out_color[pixelIndex,1] = sum_val.y;
    out_color[pixelIndex,2] = sum_val.z;

}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void process_EAWDenoise_phi(
    TensorView<float> PHI,
    uint framedim_x, uint framedim_y, int stepWidth,
    TensorView<float> occ_map,
    DiffTensorView<float> color, DiffTensorView<float> normal_map, DiffTensorView<float> pos_map,
    DiffTensorView<float> out_color
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;
    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        out_color[pixelIndex,0] = no_diff color[pixelIndex,0];
        out_color[pixelIndex,1] = no_diff color[pixelIndex,1];
        out_color[pixelIndex,2] = no_diff color[pixelIndex,2];
        return;
    }
    float3 nval = float3(normal_map[pixelIndex,0], normal_map[pixelIndex,1], normal_map[pixelIndex,2]);
    float3 pval = float3(pos_map[pixelIndex,0], pos_map[pixelIndex,1], pos_map[pixelIndex,2]);
    float3 cval = float3(color[pixelIndex,0], color[pixelIndex,1], color[pixelIndex,2]);

    float c_phi = no_diff PHI[pixelIndex,0];//0.49f;
    float n_phi = no_diff PHI[pixelIndex,1];//0.1;
    float p_phi = no_diff PHI[pixelIndex,2];//0.00003;

    float2 offset[25];
    offset[0] = float2(-2,-2);
    offset[1] = float2(-1,-2);
    offset[2] = float2(0,-2);
    offset[3] = float2(1,-2);
    offset[4] = float2(2,-2);
    
    offset[5] = float2(-2,-1);
    offset[6] = float2(-1,-1);
    offset[7] = float2(0,-1);
    offset[8] = float2(1,-1);
    offset[9] = float2(2,-1);
    
    offset[10] = float2(-2,0);
    offset[11] = float2(-1,0);
    offset[12] = float2(0,0);
    offset[13] = float2(1,0);
    offset[14] = float2(2,0);
    
    offset[15] = float2(-2,1);
    offset[16] = float2(-1,1);
    offset[17] = float2(0,1);
    offset[18] = float2(1,1);
    offset[19] = float2(2,1);
    
    offset[20] = float2(-2,2);
    offset[21] = float2(-1,2);
    offset[22] = float2(0,2);
    offset[23] = float2(1,2);
    offset[24] = float2(2,2);

    float kernel[25];
    kernel[0] = 1.0f/256.0f;
    kernel[1] = 1.0f/64.0f;
    kernel[2] = 3.0f/128.0f;
    kernel[3] = 1.0f/64.0f;
    kernel[4] = 1.0f/256.0f;
    
    kernel[5] = 1.0f/64.0f;
    kernel[6] = 1.0f/16.0f;
    kernel[7] = 3.0f/32.0f;
    kernel[8] = 1.0f/16.0f;
    kernel[9] = 1.0f/64.0f;
    
    kernel[10] = 3.0f/128.0f;
    kernel[11] = 3.0f/32.0f;
    kernel[12] = 9.0f/64.0f;
    kernel[13] = 3.0f/32.0f;
    kernel[14] = 3.0f/128.0f;
    
    kernel[15] = 1.0f/64.0f;
    kernel[16] = 1.0f/16.0f;
    kernel[17] = 3.0f/32.0f;
    kernel[18] = 1.0f/16.0f;
    kernel[19] = 1.0f/64.0f;
    
    kernel[20] = 1.0f/256.0f;
    kernel[21] = 1.0f/64.0f;
    kernel[22] = 3.0f/128.0f;
    kernel[23] = 1.0f/64.0f;
    kernel[24] = 1.0f/256.0f;
    
    float3 sum = float3(0.f);
    float cum_w = 0.0;

    [ForceUnroll]
    for(int i = 0; i < 25; i++)
    {
        uint2 uv = pixel + int2(offset[i]*stepWidth);
        if(!isValidPixel(uv, uint2(framedim_x,framedim_y))) continue;
        
        float3 ctmp = getInputElement_3(color, framedim_x, uv);
        float3 t = cval - ctmp;
        float dist2 = dot(t,t);
        float c_w = min(exp(-(dist2)/c_phi), 1.0);

        float3 ntmp = getInputElement_3(normal_map, framedim_x, uv);
        t = nval - ntmp;
        dist2 = max(dot(t,t),0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);

        float3 ptmp = getInputElement_3(pos_map, framedim_x, uv);
        t = pval - ptmp;
        dist2 = max(dot(t,t), 0.0);
        float p_w = min(exp(-(dist2)/p_phi), 1.0);

        float weight = c_w*n_w*p_w;
        sum += ctmp*weight*kernel[i];
        cum_w += weight*kernel[i];
    }

    float3 sum_val = sum/cum_w;
    out_color[pixelIndex,0] = sum_val.x;
    out_color[pixelIndex,1] = sum_val.y;
    out_color[pixelIndex,2] = sum_val.z;

}

[PreferRecompute]
float luminance(float3 v)
{
    return v.x * 0.212671f + v.y * 0.715160f + v.z * 0.072169f;
}

[PreferRecompute]
float handle_var(float3 v1, float3 v2)
{
    float3 t = v1 - v2;
    return dot(t,t);
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_variance(
    uint framedim_x, uint framedim_y, int stepWidth,
    TensorView<float> occ_map,
    TensorView<float> color, TensorView<float> normal_map, TensorView<float> pos_map,
    TensorView<float> out_phi
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;
    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if(occ_map[pixelIndex,0]<0.1f) 
    {
        out_phi[pixelIndex,0] = 0.000001f;
        out_phi[pixelIndex,1] = 0.000001f;
        out_phi[pixelIndex,2] = 0.000001f;
        return;
    }

    float c_phi = 0.000001f;
    float n_phi = 0.000001f;
    float p_phi = 0.000001f;

    float2 offset[25];
    offset[0] = float2(-2,-2);
    offset[1] = float2(-1,-2);
    offset[2] = float2(0,-2);
    offset[3] = float2(1,-2);
    offset[4] = float2(2,-2);
    
    offset[5] = float2(-2,-1);
    offset[6] = float2(-1,-1);
    offset[7] = float2(0,-1);
    offset[8] = float2(1,-1);
    offset[9] = float2(2,-1);
    
    offset[10] = float2(-2,0);
    offset[11] = float2(-1,0);
    offset[12] = float2(0,0);
    offset[13] = float2(1,0);
    offset[14] = float2(2,0);
    
    offset[15] = float2(-2,1);
    offset[16] = float2(-1,1);
    offset[17] = float2(0,1);
    offset[18] = float2(1,1);
    offset[19] = float2(2,1);
    
    offset[20] = float2(-2,2);
    offset[21] = float2(-1,2);
    offset[22] = float2(0,2);
    offset[23] = float2(1,2);
    offset[24] = float2(2,2);

    float2 sigmaVariancePair_0 = float2(0.0, 0.0);
    float2 sigmaVariancePair_1 = float2(0.0, 0.0);
    float2 sigmaVariancePair_2 = float2(0.0, 0.0);
    float2 sigmaVariancePair_n_0 = float2(0.0, 0.0);
    float2 sigmaVariancePair_n_1 = float2(0.0, 0.0);
    float2 sigmaVariancePair_n_2 = float2(0.0, 0.0);
    float2 sigmaVariancePair_p_0 = float2(0.0, 0.0);
    float2 sigmaVariancePair_p_1 = float2(0.0, 0.0);
    float2 sigmaVariancePair_p_2 = float2(0.0, 0.0);
    float sampCount = 0.0;
    float3 nval = float3(normal_map[pixelIndex,0], normal_map[pixelIndex,1], normal_map[pixelIndex,2]);
    float3 pval = float3(pos_map[pixelIndex,0], pos_map[pixelIndex,1], pos_map[pixelIndex,2]);
    float3 cval = float3(color[pixelIndex,0], color[pixelIndex,1], color[pixelIndex,2]);

    [ForceUnroll]
    for(int i = 0; i < 25; i++)
    {
        uint2 uv = pixel + int2(offset[i]*stepWidth);
        if(!isValidPixel(uv, uint2(framedim_x,framedim_y))) continue;
        
        float3 ctmp = getInputElement_3_nf(color, framedim_x, uv);
        float samp = ctmp.x;
        float sampSquared = samp * samp;
        sigmaVariancePair_0 += float2(samp, sampSquared);
        samp = ctmp.y;
        sampSquared = samp * samp;
        sigmaVariancePair_1 += float2(samp, sampSquared);
        samp = ctmp.z;
        sampSquared = samp * samp;
        sigmaVariancePair_2 += float2(samp, sampSquared);

        float3 ntmp = getInputElement_3_nf(normal_map, framedim_x, uv);
        samp = ntmp.x;
        sampSquared = samp * samp;
        sigmaVariancePair_n_0 += float2(samp, sampSquared);
        samp = ntmp.y;
        sampSquared = samp * samp;
        sigmaVariancePair_n_1 += float2(samp, sampSquared);
        samp = ntmp.z;
        sampSquared = samp * samp;
        sigmaVariancePair_n_2 += float2(samp, sampSquared);

        float3 ptmp = getInputElement_3_nf(pos_map, framedim_x, uv);
        samp = ptmp.x;
        sampSquared = samp * samp;
        sigmaVariancePair_p_0 += float2(samp, sampSquared);
        samp = ptmp.y;
        sampSquared = samp * samp;
        sigmaVariancePair_p_1 += float2(samp, sampSquared);
        samp = ptmp.z;
        sampSquared = samp * samp;
        sigmaVariancePair_p_2 += float2(samp, sampSquared);

        sampCount += 1.0;
    }

    sigmaVariancePair_0 /= sampCount;
    sigmaVariancePair_1 /= sampCount;
    sigmaVariancePair_2 /= sampCount;
    sigmaVariancePair_n_0 /= sampCount;
    sigmaVariancePair_n_1 /= sampCount;
    sigmaVariancePair_n_2 /= sampCount;
    sigmaVariancePair_p_0 /= sampCount;
    sigmaVariancePair_p_1 /= sampCount;
    sigmaVariancePair_p_2 /= sampCount;

    float c_phi_0 = max(0.0, sigmaVariancePair_0.y - sigmaVariancePair_0.x * sigmaVariancePair_0.x);
    float c_phi_1 = max(0.0, sigmaVariancePair_1.y - sigmaVariancePair_1.x * sigmaVariancePair_1.x);
    float c_phi_2 = max(0.0, sigmaVariancePair_2.y - sigmaVariancePair_2.x * sigmaVariancePair_2.x);
    c_phi = c_phi_0+c_phi_1+c_phi_2;

    float n_phi_0 = max(0.0, sigmaVariancePair_n_0.y - sigmaVariancePair_n_0.x * sigmaVariancePair_n_0.x);
    float n_phi_1 = max(0.0, sigmaVariancePair_n_1.y - sigmaVariancePair_n_1.x * sigmaVariancePair_n_1.x);
    float n_phi_2 = max(0.0, sigmaVariancePair_n_2.y - sigmaVariancePair_n_2.x * sigmaVariancePair_n_2.x);
    n_phi = n_phi_0+n_phi_1+n_phi_2;

    float p_phi_0 = max(0.0, sigmaVariancePair_p_0.y - sigmaVariancePair_p_0.x * sigmaVariancePair_p_0.x);
    float p_phi_1 = max(0.0, sigmaVariancePair_p_1.y - sigmaVariancePair_p_1.x * sigmaVariancePair_p_1.x);
    float p_phi_2 = max(0.0, sigmaVariancePair_p_2.y - sigmaVariancePair_p_2.x * sigmaVariancePair_p_2.x);
    p_phi = p_phi_0+p_phi_1+p_phi_2;

    out_phi[pixelIndex,0] = 2.0f*c_phi;
    out_phi[pixelIndex,1] = 0.1f*n_phi;
    out_phi[pixelIndex,2] = 0.1f*p_phi;
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_normal_ao(
    uint framedim_x, uint framedim_y,
    TensorView<float> occ_map,
    TensorView<float> normal_map, TensorView<float> ray_dir,
    TensorView<float> out_ao
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= framedim_x || dispatchIdx.y >= framedim_y) return;
    uint2 pixel = uint2(dispatchIdx.x, dispatchIdx.y);
    uint pixelIndex = dispatchIdx.y * framedim_x + dispatchIdx.x;
    if (occ_map[pixelIndex, 0] < 0.1f)
    {
        out_ao[pixelIndex, 0] = 0;
        out_ao[pixelIndex, 1] = 0;
        out_ao[pixelIndex, 2] = 0;
        return;
    }
    float3 nval = float3(normal_map[pixelIndex, 0], normal_map[pixelIndex, 1], normal_map[pixelIndex, 2]);
    //float3 view_dir = float3(ray_dir[pixelIndex, 0], ray_dir[pixelIndex, 1], ray_dir[pixelIndex, 2]);
    //view_dir = -view_dir;


    int count = 0;
    float sum_val = 0.f;
    float sum = 0.f;
    int width = 4;
    [ForceUnroll]
    for (int i = -width; i < width; i++)
        for (int j = -width; j < width; j++)
    {
        uint2 uv = pixel + int2(i,j);
        if (!isValidPixel(uv, uint2(framedim_x, framedim_y))) continue;
        if (occ_map[uv.y * framedim_x + uv.x, 0] < 0.1f) continue;

        float3 ntmp = getInputElement_3_nf(normal_map, framedim_x, uv);
        // float3 t = nval - ntmp;
        float dist2 = max(dot(ntmp, nval), 0.0);
        dist2 = min(1.0f, dist2);
        sum += dist2;
        count++;
    }

    float normal_weight = 1 - sum / count;
    //if (normal_weight < 0.1) normal_weight = 0.2;
    //else normal_weight = 1.0;

    //sum_val = max(dot(nval, view_dir), 0);

    sum_val = normal_weight * 50;
    sum_val = clamp(sum_val, 0, 1);

    //float sum_val = max(dot(nval, view_dir),0);
    out_ao[pixelIndex, 0] = sum_val;
    out_ao[pixelIndex, 1] = sum_val;
    out_ao[pixelIndex, 2] = sum_val;
}