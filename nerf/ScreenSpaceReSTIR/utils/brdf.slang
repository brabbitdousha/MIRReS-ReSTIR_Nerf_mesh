float3 perp_stark(float3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    float3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym);  // 1 ^ (xm & ym)
    float3 v = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}

float3 toLocal(float3 w, float3 N)
{
    float3 B = perp_stark(N);
    float3 T = cross(B, N);
    return float3(dot(B, w), dot(T, w), dot(N, w));
}

[PreferRecompute]
float3 evalFresnelSchlick(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

[PreferRecompute]
float evalFresnelSchlick(float f0, float f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

[PreferRecompute]
float evalNdfGGX(float alpha, float cosTheta)
{
    const float M_PI = 3.141592653589793f;
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * M_PI);
}

float evalMaskingSmithGGXSeparable(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / ((1 + lambdaI) * (1 + lambdaO));
}

[PreferRecompute]
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}

float evalPdfGGX_NDF(float alpha, float cosTheta)
{
    return evalNdfGGX(alpha, cosTheta) * cosTheta;
}

float2 sample_disk_concentric(float2 u)
{
    const float M_PI_4 = 0.785398163397448309616f;
    const float M_PI_2 = 1.57079632679489661923f;

    u = 2.f * u - 1.f;
    if (u.x == 0.f && u.y == 0.f) return u;
    float phi, r;
    if (abs(u.x) > abs(u.y))
    {
        r = u.x;
        phi = (u.y / u.x) * M_PI_4;
    }
    else
    {
        r = u.y;
        phi = M_PI_2 - (u.x / u.y) * M_PI_4;
    }
    return r * float2(cos(phi), sin(phi));
}

float3 sample_cosine_hemisphere_concentric(float2 u, out float pdf)
{
    const float M_1_PI = 0.31830988f;
    float2 d = sample_disk_concentric(u);
    float z = sqrt(max(0.f, 1.f - dot(d, d)));
    pdf = z * M_1_PI;
    return float3(d, z);
}

float3 toGlobal(float3 w, float3 N)
{
    float3 B = perp_stark(N);
    float3 T = cross(B, N);
    return B * w.x + T * w.y + N * w.z;
}

float3 sampleGGX_NDF(float alpha, float2 u, out float pdf)
{
    const float M_PI = 3.141592653589793f;
    float alphaSqr = alpha * alpha;
    float phi = u.y * (2 * M_PI);
    float tanThetaSqr = alphaSqr * u.x / (1 - u.x);
    float cosTheta = 1 / sqrt(1 + tanThetaSqr);
    float r = sqrt(max(1 - cosTheta * cosTheta, 0));

    pdf = evalPdfGGX_NDF(alpha, cosTheta);
    return float3(cos(phi) * r, sin(phi) * r, cosTheta);
}

[PreferRecompute]
float3 DiffuseReflection_eval(float3 wo, float3 wi, float3 albedo)
{
    const float M_1_PI = 0.31830988f;
    if (min(wo.z, wi.z) < 1e-6f) return float3(0.f);

    return M_1_PI * albedo * wi.z;
}

[PreferRecompute]
float3 SpecularReflection_eval(float3 wo, float3 wi, float3 albedo, float alpha, bool activeLobes = false, bool allowDeltaEval = false)
{
    if (min(wo.z, wi.z) < 1e-6f) return float3(0.f);

    // Handle delta reflection.
    if (alpha == 0.f) //return float3(0.f);
    {
        //print("delta!",1);

        if (allowDeltaEval) return evalFresnelSchlick(albedo, 1.f, wo.z);
        else return float3(0.f);
    }
    if(!activeLobes) return float3(0.f);

    float3 h = normalize(wo + wi);
    float woDotH = dot(wo, h);
    float D = evalNdfGGX(alpha, h.z);
    float G = evalMaskingSmithGGXCorrelated(alpha, wo.z, wi.z);
    float3 F = evalFresnelSchlick(albedo, 1, woDotH);
    return F * D * G * 0.25f / wo.z;
}

float evalBRDF(float3 L, float3 V, float3 N, float ggxAlpha, float diffuseWeight, float specularWeight)
{
    const float M_1_PI = 0.31830988f;
    float weightSum = diffuseWeight + specularWeight;
    float diffuseSpecularMix = weightSum > 1e-7f ? (diffuseWeight / weightSum) : 1.f;

    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));

    float D = evalNdfGGX(ggxAlpha, NdotH);
    float G = evalMaskingSmithGGXSeparable(ggxAlpha, NdotV, NdotL);
    float F = specularWeight < 1e-8f ? 0.f : evalFresnelSchlick(specularWeight, 1.f, LdotH) / specularWeight;

    float diffuse = NdotL * M_1_PI;
    float specular = max(0.f, D * G * F / (4.f * NdotV));
    return NdotL > 0.f ? lerp(specular, diffuse, diffuseSpecularMix) : 0.f;
}

float evalPdfBRDF(bool specularOnly, float3 dir, float3 V, float3 N, float ggxAlpha, float diffuseWeight, float specularWeight)
{
    const float M_1_PI = 0.31830988f;
    float weightSum = diffuseWeight + specularWeight;
    float diffuseSpecularMix = weightSum > 1e-7f ? (diffuseWeight / weightSum) : 1.f;

    float cosTheta = saturate(dot(N, dir));
    float diffusePdf = specularOnly ? 0.f : cosTheta * M_1_PI;
    float3 h = normalize(toLocal(dir + V, N));
    // float specularPdf = evalPdfGGX_VNDF(ggxAlpha, toLocal(dir), h) / (4.f * saturate(dot(h, toLocal(V))));
    float specularPdf = evalPdfGGX_NDF(ggxAlpha, h.z) / (4.f * saturate(dot(h, toLocal(V, N))));
    float pdf = cosTheta > 0.f ? lerp(specularPdf, diffusePdf, diffuseSpecularMix) : 0.f;
    return pdf;
}

bool sampleBRDF(bool specularOnly, float3 xi, out float3 dir, float3 V, float3 N, float ggxAlpha, float diffuseWeight, float specularWeight)
{
    float weightSum = diffuseWeight + specularWeight;
    float diffuseSpecularMix = weightSum > 1e-7f ? (diffuseWeight / weightSum) : 1.f;
    dir = {};

    if (xi.x < diffuseSpecularMix)
    {
        if (specularOnly) return false;
        float pdf;
        dir = toGlobal(sample_cosine_hemisphere_concentric(xi.yz, pdf), N);
    }
    else
    {
        float pdf;
        // float3 h = sampleGGX_VNDF(ggxAlpha, toLocal(V), xi.yz, pdf);
        float3 h = sampleGGX_NDF(ggxAlpha, xi.yz, pdf);
        dir = reflect(-V, toGlobal(h, N));
    }

    return dot(N, dir) > 0.f;
}