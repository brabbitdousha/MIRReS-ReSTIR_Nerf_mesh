struct Frame : IDifferentiable
{
	float3 o;
	float3 x;
	float3 y;
	float3 z;
};

[Differentiable]
[PreferRecompute]
float copysignf(float a, float b){
	if(b>0) return a;
	else return -a;
}

[Differentiable]
[PreferRecompute]
Frame create_frame(float3 normal){
	Frame frame;
	frame.o = float3(0,0,0);
	frame.z = normal;
	float sign = copysignf(1.0f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    frame.x = float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    frame.y = float3(b, sign + normal.y * normal.y * a, -normal.y);
	return frame;
}

[Differentiable]
[PreferRecompute]
void frame_to_local(Frame frame, inout float3 v){
    v = float3(dot(frame.x,v), dot(frame.y, v), dot(frame.z, v));
}

[Differentiable]
[PreferRecompute]
void frame_to_global(Frame frame, inout float3 v){
	v = frame.x * v.x + frame.y * v.y + frame.z * v.z;
}

[PreferRecompute]
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
float3 eval_bi(TensorView<float> env_tex, float2 uv, int width, int height)
{
    // bilinear interpolation
    float x = uv.x * width - 0.5f;
    float y = uv.y * height - 0.5f;
    int x0 = x < 0.f ? int(x) - 1 : int(x);
    int y0 = y < 0.f ? int(y) - 1 : int(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
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

[Differentiable]
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

#define MAX_STACK_SIZE 64

struct GPUBvhNode {
    int index;
    int left;          // pointer to the left child or INVALID_POINTER in case of leaf
    int right;         // pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node
};

/*
intersection code from https://github.com/maxilevi/raytracer/tree/master
*/
[PreferRecompute]
bool aabb_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 bbox_min, float3 bbox_max)
{
    for (int i = 0; i < 3; ++i)
    {
        float ray_d_i = rayd[i];
        if (ray_d_i == 0.f) ray_d_i = 0.000001f;
        float inverse_direction = 1.0 / ray_d_i;
        float t0 = (bbox_min[i] - rayo[i]) * inverse_direction;
        float t1 = (bbox_max[i] - rayo[i]) * inverse_direction;
        if (inverse_direction < 0.0) {
            float tmp = t1;
            t1 = t0;
            t0 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;

}
[PreferRecompute]
bool triangle_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos)
{
    float epsilon = 1e-15;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;
    return true;
}
[PreferRecompute]
bool bvh_hit(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos)
{
    rayd = normalize(rayd);
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            bool hit = triangle_hit(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
    }
    return any_hit;
}

[PreferRecompute]
bool triangle_hit_with_normal(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos, inout float3 normal)
{
    float epsilon = 1e-15;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;

    float3 face_normal = normalize(cross(E1, E2));
    float r = 1.0 - u - v;
    normal = u * face_normal + v * face_normal + r * face_normal;
    if(dot(-rayd, normal)<0)
    {
        normal = -normal;
    }
    normal = normalize(normal);

    return true;
}

[PreferRecompute]
bool bvh_hit_with_normal(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos, inout float3 normal)
{
    rayd = normalize(rayd);
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);
    float3 hit_normal = float3(1.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            float3 now_normal_hit = float3(1.f);
            bool hit = triangle_hit_with_normal(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit, now_normal_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
                if(now_t_hit<=closest_so_far)
                    hit_normal = now_normal_hit;  
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
        normal = hit_normal;
    }
    return any_hit;
}

[PreferRecompute]
bool IsBlack(float3 v)
{
    if (v.x != 0.) return false;
    if (v.y != 0.) return false;
    if (v.z != 0.) return false;

    return true;
}

[PreferRecompute]
float power_heuristic(float p1, float p2) {
    return p1 * p1 / (p1 * p1 + p2 * p2);
}