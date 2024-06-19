import random;
import helper;
struct PresampledLight {
    float3 light_data;
    int2 light_uv;
    float inv_pdf;
};
[PreferRecompute]
PresampledLight createInvalid_PresampledLight()
{
    PresampledLight ls;
    ls.light_data = float3(0.f);
    ls.light_uv = int2(0,0);
    ls.inv_pdf = 0.f;
    return ls;
}
[PreferRecompute]
int FindInterval_mcdf_(int left,int right,float val, TensorView<float> mcdf_)
{
	int l=left,r=right;
	int mid;
	while(l<r)
	{
		mid=(l+r)/2;
		if(mcdf_[mid]<=val) l=mid+1;
		else r=mid;
	}
	return clamp(l-left-1,0,right-left);
}
[PreferRecompute]
int FindInterval_cdf_(int left,int right,float val,TensorView<float> cdf_)
{
	int l=left,r=right;
	int mid;
	while(l<r)
	{
		mid=(l+r)/2;
		if(cdf_[mid]<=val) l=mid+1;
		else r=mid;
	}
	return clamp(l-left-1,0,right-left);
}
[PreferRecompute]
void warp(inout float2 uv, inout int row, inout int col, int w_, int h_,
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_)
{
    row = FindInterval_mcdf_(0, mcdf_.size(0), uv.y, mcdf_);
    uv.y = clamp((uv.y - mcdf_[row]) / mpdf_[row], 0.0f, 1.0f);
    int row_start = row * (w_ + 1);
    int row_end = row_start + (w_ + 1);
    col = FindInterval_cdf_(row_start,row_end,uv.x, cdf_);
    int ic = row * (w_ + 1) + col;
    int ip = row * w_ + col;
    uv.x = clamp((uv.x - cdf_[ic]) / pdf_[ip], 0.0f, 1.0f);
}
[PreferRecompute]
void warp_continue(inout float2 uv, inout int row, inout int col, int w_, int h_,
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_)
{
    warp(uv,row,col,w_,h_,pdf_, cdf_, mpdf_, mcdf_);
    uv.x = clamp((uv.x + col) / w_, 0.0f, 1.0f);
    uv.y = clamp((uv.y + row) / h_, 0.0f, 1.0f);
}
[PreferRecompute]
float pdf(inout int row,inout int col,int w_, int h_, TensorView<float> pdf_, TensorView<float> mpdf_) 
{
    row = clamp(row, 0, h_ - 1);
    col = clamp(col, 0, w_ - 1);
    return pdf_[row * w_ + col] * mpdf_[row];
}
[PreferRecompute]
float pdf_continue(inout int row,inout int col, int w_, int h_, TensorView<float> pdf_, TensorView<float> mpdf_)
{
    return pdf(row, col, w_, h_, pdf_, mpdf_) * w_ * h_;
}
[PreferRecompute]
float4 scene_env_alias_table(TensorView<float> env_tex, int2 uv, uint width, uint height)
{   
    int x0 = uv.x;
    int y0 = uv.y;
    return float4(env_tex[get_data_idx(x0, y0, width), 0],env_tex[get_data_idx(x0, y0, width), 1],
    env_tex[get_data_idx(x0, y0, width), 2], env_tex[get_data_idx(x0, y0, width), 3]);
}
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
bool InfiniteAreaLight_Sample_Li(TensorView<float> env_tex, uint width, uint height, float2 rnd_val, 
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_,
    out float3 dir, out float3 weight, out float out_pdf, out int2 light_uv){
    float2 uv=rnd_val;
    const float PI = 3.141592653589793f;
    int row=0, col=0;
    warp_continue(uv, row, col, width, height, pdf_, cdf_, mpdf_, mcdf_);
    float pdf = pdf_continue(row, col, width, height, pdf_, mpdf_);
    float theta = uv.y * PI, phi = uv.x * 2 * PI;
    float cos_theta = cos(theta), cos_phi = cos(phi), sin_theta = sin(theta),
          sin_phi = sin(phi);
    
    dir = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);

    if(abs(sin_theta) >= 1e-4)
        pdf = pdf / (2 * PI * PI * sin_theta);
    else
        pdf = 0.0f;

    out_pdf=pdf;

    //weight = scene_env_tex_sample(env_tex, float2(uv.x, 1 - uv.y), width, height)/pdf;
    float3 temp_dir = dir;
    float3 r_dir = ngp_dir(temp_dir);
    weight = env_le(r_dir, env_tex, width, height)/pdf;

    light_uv = uv2xy(float2(uv.x, 1 - uv.y), width, height);

    if(pdf == 0)
        return false;
    return true;

}

//alias_map's z is in env_tex order
[PreferRecompute]
bool env_sample_li(TensorView<float> env_tex, uint width, uint height, float2 rnd_val, 
    TensorView<float> alias_map,
    out float3 dir, out float3 weight, out float out_pdf, out int2 light_uv){
    const float TWO_PI = 6.2831853f;
    const float INV_PI = 0.31830988f;
    const float PI = 3.141592653589793f;
    const int size = width * height;
    const int idx  = min(int(rnd_val.x * float(size)), size - 1);
    rnd_val.x = rnd_val.x * float(size) - idx;
    const int px = idx % width;
    const int py = idx / width;
    // vec4 alias_data = texelFetch(alias_map,ivec2(px,tsize.y-py-1), 0).xyzw;
    float4 alias_data = scene_env_alias_table(alias_map, int2(px, height - py - 1), width, height);
    float q = alias_data.x;
    int env_idx;
    if(rnd_val.x > q){
       env_idx = int(alias_data.y);
       rnd_val.x = (rnd_val.x-q) / (1-q);
       out_pdf = alias_data.w;
    }
    else{
        env_idx = idx;
        rnd_val.x = rnd_val.x / q;
        out_pdf = alias_data.z;
    }
    const int ex = env_idx % width;
    const int ey = env_idx / width;
    float eu = (ex + rnd_val.x) / width;
    float phi = eu * TWO_PI;
    const float step_theta = PI / float(height);
    const float theta0     = float(ey) * step_theta;
    const float cos_theta  = cos(theta0) * rnd_val.y + cos(theta0 + step_theta) * (1-rnd_val.y);
    const float theta      = acos(cos_theta);
    float cos_phi = cos(phi), sin_theta = sin(theta),
          sin_phi = sin(phi);
    dir = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
    float ev = theta * INV_PI;
    weight = scene_env_tex_sample(env_tex, float2(eu, 1 - ev), width, height) / out_pdf;
    light_uv = uv2xy(float2(eu, 1 - ev), width, height);
    if(out_pdf == 0)
        return false;
    return true;

}

[PreferRecompute]
float AliasTable_sample(uint index, float rnd, TensorView<float> items){
    float Threshold = items[index,0];
    float IndexA = items[index,1];
    float IndexB = items[index,2];
    return (rnd >= Threshold) ? IndexA : IndexB;
}

[PreferRecompute]
bool envLightAliasTable_sample(TensorView<float> env_tex, uint width, uint height, float2 rnd_val, 
    TensorView<float> items, float weightSum,
    out float3 dir, out float3 weight, out float out_pdf, out int2 light_uv){
    const float PI = 3.141592653589793f;
    uint count = width*height;
    uint index = min(count - 1, (uint)(rnd_val.x * count));
    float sampled_index = AliasTable_sample(index, rnd_val.y, items);
    weight = 0.f;

    const int px = int(sampled_index) % width;
    const int py = int(sampled_index) / width;

    out_pdf = luminance(eval_bi(env_tex,float2((px + .5f) / width, (py + .5f) / height), width, height))/weightSum;
    float theta = (float(py)/height) * PI, phi = (float(px)/width) * 2 * PI;
    float cos_theta = cos(theta), cos_phi = cos(phi), sin_theta = sin(theta),
          sin_phi = sin(phi);
    
    dir = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
    light_uv = uv2xy(float2((px + .5f) / width, (py + .5f) / height), width, height);

    if(out_pdf == 0)
        return false;
    return true;
}

[PreferRecompute]
PresampledLight presampled_light(float random, inout uint seed, TensorView<float> env_tex, uint width, uint height,
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_)
{
    PresampledLight pls = createInvalid_PresampledLight();
    float3 samp_dir, samp_weight;
    float samp_pdf;
    int2 light_uv;
    bool res = InfiniteAreaLight_Sample_Li(env_tex, width, height, float2(sampleNext1D(seed), sampleNext1D(seed)), 
    pdf_, cdf_, mpdf_, mcdf_,
    samp_dir, samp_weight, samp_pdf, light_uv);
    if (res)
    {
        pls.light_data.x = 1.f;
        float2 real_light_uv = oct_encode(samp_dir);
        pls.light_data.y = real_light_uv.x;
        pls.light_data.z = real_light_uv.y;
        pls.light_uv = light_uv;
        pls.inv_pdf = samp_pdf;
    }
    return pls;
}
[PreferRecompute]
PresampledLight presampled_light_alias(float random, inout uint seed, TensorView<float> env_tex, uint width, uint height,
    TensorView<float> alias_map)
{
    PresampledLight pls = createInvalid_PresampledLight();
    float3 samp_dir, samp_weight;
    float samp_pdf;
    int2 light_uv;
    bool res = env_sample_li(env_tex, width, height, float2(sampleNext1D(seed), sampleNext1D(seed)), 
    alias_map,
    samp_dir, samp_weight, samp_pdf, light_uv);
    if (res)
    {
        pls.light_data.x = 1.f;
        float2 real_light_uv = oct_encode(samp_dir);
        pls.light_data.y = real_light_uv.x;
        pls.light_data.z = real_light_uv.y;
        pls.light_uv = light_uv;
        pls.inv_pdf = samp_pdf;
    }
    return pls;
}
[PreferRecompute]
PresampledLight sampleEnv(float random, inout uint seed, TensorView<float> env_tex, uint width, uint height,
    TensorView<float> items, float weightSum)
{
    PresampledLight pls = createInvalid_PresampledLight();
    float3 samp_dir, samp_weight;
    float samp_pdf;
    int2 light_uv;
    bool res = envLightAliasTable_sample(env_tex, width, height, float2(random, sampleNext1D(seed)), 
    items, weightSum,
    samp_dir, samp_weight, samp_pdf, light_uv);
    if (res)
    {
        pls.light_data.x = 1.f;
        float2 real_light_uv = oct_encode(samp_dir);
        pls.light_data.y = real_light_uv.x;
        pls.light_data.z = real_light_uv.y;
        pls.light_uv = light_uv;
        pls.inv_pdf = samp_pdf;
    }
    return pls;
}

[PreferRecompute]
float3 ngp_dir(float3 curr_ray_dir)
{
    float3 new_dir = float3(-curr_ray_dir.x, curr_ray_dir.z, curr_ray_dir.y);
    return new_dir;
}