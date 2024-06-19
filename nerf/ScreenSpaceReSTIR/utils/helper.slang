[PreferRecompute]
[Differentiable]
float3 math_lerp(float3 t00, float3 t01, float3 t10, float3 t11, float2 uv)
{
    return (t00 * (1.0f - uv.x) + t01 * uv.x) * (1.0f - uv.y)
        + (t10 * (1.0f - uv.x) + t11 * uv.x) * uv.y;
}
[PreferRecompute]
int get_data_idx(int x, int y, int width)
{
        return y * width + x;
}
[PreferRecompute]
float3 eval_near(TensorView<float> env_tex, float2 uv, int width, int height)
{
    // nearest neightbor interpolation
    float x = uv.x * width;
    float y = uv.y * height;
    int x0 = x < 0.f ? int(x) - 1 : int(x);
    int y0 = y < 0.f ? int(y) - 1 : int(y);
    x0 = ((x0 % width) + width) % width;
    y0 = ((y0 % height) + height) % height;
    return float3(env_tex[get_data_idx(x0, y0, width), 0],env_tex[get_data_idx(x0, y0, width), 1],env_tex[get_data_idx(x0, y0, width), 2]);
}

[PreferRecompute]
int2 uv2xy(float2 uv, int width, int height)
{
    // nearest neightbor interpolation
    float x = uv.x * width;
    float y = uv.y * height;
    int x0 = x < 0.f ? int(x) - 1 : int(x);
    int y0 = y < 0.f ? int(y) - 1 : int(y);
    x0 = ((x0 % width) + width) % width;
    y0 = ((y0 % height) + height) % height;
    return int2(x0,y0);
}

[PreferRecompute]
float3 eval_env(TensorView<float> env_tex, int2 uv, int width, int height)
{
   return float3(env_tex[get_data_idx(uv.x, uv.y, width)]);
}

[PreferRecompute]
float3 eval_bi(TensorView<float> env_tex, float2 uv, int width, int height)
{
    // bilinear interpolation
    float x = uv.x * width - 0.5f;
    float y = uv.y * height - 0.5f;
    int x0 = int(x);
    int y0 = int(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float u = x - x0;
    float v = y - y0;
    x0 = ((x0 % width) + width) % width;
    y0 = ((y0 % height) + height) % height;
    x1 = ((x1 % width) + width) % width;
    y1 = ((y1 % height) + height) % height;
    return math_lerp(float3(env_tex[get_data_idx(x0, y0, width), 0],env_tex[get_data_idx(x0, y0, width), 1],env_tex[get_data_idx(x0, y0, width), 2]),
     float3(env_tex[get_data_idx(x1, y0, width), 0],env_tex[get_data_idx(x1, y0, width), 1],env_tex[get_data_idx(x1, y0, width), 2]), 
     float3(env_tex[get_data_idx(x0, y1, width), 0],env_tex[get_data_idx(x0, y1, width), 1],env_tex[get_data_idx(x0, y1, width), 2]), 
     float3(env_tex[get_data_idx(x1, y1, width), 0],env_tex[get_data_idx(x1, y1, width), 1],env_tex[get_data_idx(x1, y1, width), 2]), 
     float2(u, v));
}
[PreferRecompute]
[Differentiable]
float3 eval_bi_di(DiffTensorView<float> env_tex, float2 uv, int width, int height)
{
    // bilinear interpolation
    float x = uv.x * width - 0.5f;
    float y = uv.y * height - 0.5f;
    int x0 = int(x);
    int y0 = int(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float u = x - x0;
    float v = y - y0;
    x0 = ((x0 % width) + width) % width;
    y0 = ((y0 % height) + height) % height;
    x1 = ((x1 % width) + width) % width;
    y1 = ((y1 % height) + height) % height;
    return math_lerp(float3(env_tex[get_data_idx(x0, y0, width), 0],env_tex[get_data_idx(x0, y0, width), 1],env_tex[get_data_idx(x0, y0, width), 2]),
     float3(env_tex[get_data_idx(x1, y0, width), 0],env_tex[get_data_idx(x1, y0, width), 1],env_tex[get_data_idx(x1, y0, width), 2]), 
     float3(env_tex[get_data_idx(x0, y1, width), 0],env_tex[get_data_idx(x0, y1, width), 1],env_tex[get_data_idx(x0, y1, width), 2]), 
     float3(env_tex[get_data_idx(x1, y1, width), 0],env_tex[get_data_idx(x1, y1, width), 1],env_tex[get_data_idx(x1, y1, width), 2]), 
     float2(u, v));
}
[PreferRecompute]
float luminance(float3 v)
{
    return v.x * 0.212671f + v.y * 0.715160f + v.z * 0.072169f;
}
[PreferRecompute]
float2 oct_wrap(float2 v) 
{
    return (1.0 - abs(float2(v.y, v.x))) * float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}
[PreferRecompute]
float2 oct_encode(float3 n) 
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 n_xy = float2(n.x, n.y);
    n_xy = oct_wrap(n_xy);
    n.x = n.z >= 0.0 ? n.x : n_xy.x;
    n.y = n.z >= 0.0 ? n.y : n_xy.y;
    n = n * 0.5 + 0.5; //n.xy = n.xy * 0.5 + 0.5;
    return float2(n.x,n.y);
}
[PreferRecompute]
float3 oct_decode(float2 f) 
{
    f = f * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    float2 temp = float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    n.x += temp.x;
    n.y += temp.y;
    return normalize(n);
}