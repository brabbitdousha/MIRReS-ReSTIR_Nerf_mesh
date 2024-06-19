import random;

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

[Differentiable]
[PreferRecompute]
float3 evalFresnelSchlick(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

[Differentiable]
[PreferRecompute]
float evalFresnelSchlick(float f0, float f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

[Differentiable]
[PreferRecompute]
float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}
[Differentiable]
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

[Differentiable]
[PreferRecompute]
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}

[PreferRecompute]
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

[PreferRecompute]
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

[Differentiable]
[PreferRecompute]
float3 DiffuseReflection_eval(float3 wo, float3 wi, float3 albedo)
{
    const float M_1_PI = 0.31830988f;
    if (min(wo.z, wi.z) < 1e-6f) return float3(0.f);

    return M_1_PI * albedo * wi.z;
}

[PreferRecompute]
float DiffuseReflection_evalPdf(float3 wo, float3 wi)
{
    const float M_1_PI = 0.31830988f;
    if (min(wo.z, wi.z) < 1e-6f) return 0.f;

    return M_1_PI * wi.z;
}

[PreferRecompute]
bool DiffuseReflection_sample(float3 albedo, no_diff float3 wo,no_diff out float3 wi, no_diff out float pdf, inout float3 weight, no_diff inout uint sg)
{
    sampleNext1D(sg);
    wi = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);

    if (min(wo.z, wi.z) < 1e-6f)
    {
        weight = float3(0.f);
        return false;
    }

    weight = albedo;
    return true;
}

[Differentiable]
[PreferRecompute]
float3 Diffuse_light(float3 wo, float3 wi)
{
    const float M_1_PI = 0.31830988f;
    if (min(wo.z, wi.z) < 1e-6f) return float3(0.f);

    return max(M_1_PI * wi.z, 0.0f);
}

[Differentiable]
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

[PreferRecompute]
float SpecularReflection_evalPdf(float3 wo, float3 wi, float alpha, bool activeLobes = false, bool allowDeltaEval = false)
{
    if (min(wo.z, wi.z) < 1e-6f) return 0.f;

    // Handle delta reflection.
    if (alpha == 0.f) // return float3(0.f);
    {
        // print("delta!",1);

        if (allowDeltaEval) return 1.0f;
        else return 0.f;
    }
    if (!activeLobes) return 0.f;

    float3 h = normalize(wo + wi);
    float woDotH = dot(wo, h);

    float pdf = evalPdfGGX_NDF(alpha, h.z);

    return pdf / (4.f * woDotH);
}

[PreferRecompute]
bool SpecularReflection_sample(float3 albedo, float alpha, float3 wo, out float3 wi, out float pdf, inout float3 weight, inout uint sg,
    bool activeLobes = false, bool allowDeltaEval = false)
{
    // Default initialization to avoid divergence at returns.
    wi = {};
    weight = float3(0.f);
    pdf = 0.f;
    //lobe = (uint)LobeType::SpecularReflection;
    if (wo.z < 1e-6f) return false;

    sampleNext1D(sg);

    if (alpha == 0.f)
    {
        if (!allowDeltaEval) return false;
        sampleNext2D(sg);
        wi = float3(-wo.x, -wo.y, wo.z);
        pdf = 1.f; // 0.f;
        weight = evalFresnelSchlick(albedo, 1.f, wo.z);
        return true;
    }

    if (!activeLobes) return false;
    // Sample the GGX distribution to find a microfacet normal (half vector).

    //float3 h = sampleGGX_VNDF(alpha, wo, sampleNext2D(sg), pdf); // pdf = G1(wo) * D(h) * max(0,dot(wo,h)) / wo.z
    float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf); // pdf = D(h) * h.z

    // Reflect the outgoing direction to find the incident direction.
    float woDotH = dot(wo, h);
    wi = 2.f * woDotH * h - wo;
    if (wi.z < 1e-6f) return false;
    // For testing.
    pdf = SpecularReflection_evalPdf(wo, wi, alpha, activeLobes, allowDeltaEval);
    weight = SpecularReflection_eval(wo, wi, albedo, alpha, activeLobes, allowDeltaEval) / pdf;
    //lobe = (uint)LobeType::SpecularReflection;
    return true;
}

[Differentiable]
[PreferRecompute]
float3 FalcorBRDF_eval(float pDiffuseReflection, float pSpecularReflection, float alpha, float3 spec_albedo, float3 diff_albedo,
                       float3 wo, float3 wi, bool activeLobes = false, bool allowDeltaEval = false)
{
    float3 result = 0.f;
    if (pDiffuseReflection > 0.f && !allowDeltaEval) result += DiffuseReflection_eval(wo, wi, diff_albedo);
    if (pSpecularReflection > 0.f) result += SpecularReflection_eval(wo, wi, spec_albedo, alpha, activeLobes, allowDeltaEval);
    return result;
}

[PreferRecompute]
float FalcorBRDF_evalPdf(float pDiffuseReflection, float pSpecularReflection, float3 wo, float3 wi, float alpha, bool activeLobes = false, bool allowDeltaEval = false)
{
    float pdf = 0.f;
    if (!allowDeltaEval && pDiffuseReflection > 0.f) pdf += pDiffuseReflection * DiffuseReflection_evalPdf(wo, wi);
    if (pSpecularReflection > 0.f) pdf += pSpecularReflection * SpecularReflection_evalPdf(wo, wi, alpha, activeLobes, allowDeltaEval);
    return pdf;
}

[PreferRecompute]
bool FalcorBRDF_sample(float pDiffuseReflection, float pSpecularReflection,
                        float3 wo, out float3 wi, out float pdf, inout uint specularBounce, inout float3 weight, inout uint sg,
    float alpha, float3 spec_albedo, float3 diff_albedo, bool activeLobes = false, bool allowDeltaEval = false)
{
    wi = {};
    weight = float3(0.f);
    pdf = 0.f;
    specularBounce = 0;

    bool valid = false;
    float uSelect = sampleNext1D(sg);

    if (uSelect < pDiffuseReflection)
    {
        valid = DiffuseReflection_sample(diff_albedo, wo, wi, pdf, weight, sg);
        //weight /= pDiffuseReflection;
        weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
                                 wo, wi, true, false);
        pdf *= pDiffuseReflection;
        // if (pDiffuseTransmission > 0.f) pdf += pDiffuseTransmission * diffuseTransmission.evalPdf(wo, wi);
        if (pSpecularReflection > 0.f) pdf += pSpecularReflection * SpecularReflection_evalPdf(wo, wi, alpha, activeLobes, allowDeltaEval);

        weight /= pdf;
    }
    else if (uSelect < pDiffuseReflection + pSpecularReflection)
    {
        valid = SpecularReflection_sample(spec_albedo, alpha, wo, wi, pdf, weight, sg, activeLobes, allowDeltaEval);
        //weight /= pSpecularReflection;
        weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
                                 wo, wi, true, false);
        pdf *= pSpecularReflection;
        float test_roughness = sqrt(alpha);
        if (test_roughness > 0.15f )
        {
            if (pDiffuseReflection > 0.f) pdf += pDiffuseReflection * DiffuseReflection_evalPdf(wo, wi);
        }
        else
        {
            specularBounce = 1;
        }
        weight /= pdf;
    }
    return valid;
}

[Differentiable]
[PreferRecompute]
bool FalcorBRDF_sample_se(float pDiffuseReflection, float pSpecularReflection,
                        float3 wo, out float3 wi, no_diff out float pdf, no_diff inout uint specularBounce, inout float3 weight, inout float3 diff_weight, inout float3 spec_weight, no_diff inout uint sg,
    float alpha, float3 spec_albedo, float3 diff_albedo, bool activeLobes = false, bool allowDeltaEval = false)
{
    wi = {};
    weight = float3(0.f);
    pdf = 0.f;
    specularBounce = 0;

    float3 diff_f = float3(0.0, 0.0, 0.0);
    float3 spec_f = float3(0.0, 0.0, 0.0);

    bool valid = false;
    float uSelect = sampleNext1D(sg);

    if (uSelect < pDiffuseReflection)
    {
        valid = DiffuseReflection_sample(diff_albedo, wo, wi, pdf, weight, sg);

        //weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
        //                         wo, wi, true, false);
        if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wo, wi);
        if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wo, wi, spec_albedo, alpha, true);
        weight  = diff_albedo*diff_f + spec_f;
                        
        pdf *= detach(pDiffuseReflection);
        if (pSpecularReflection > 0.f) pdf += detach(pSpecularReflection) * SpecularReflection_evalPdf(wo, wi, alpha, activeLobes, allowDeltaEval);

        weight /= pdf;
        diff_f /= pdf;
        spec_f /= pdf;
    }
    else if (uSelect < pDiffuseReflection + pSpecularReflection)
    {
        valid = SpecularReflection_sample(spec_albedo, alpha, wo, wi, pdf, weight, sg, activeLobes, allowDeltaEval);
        //weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
        //                         wo, wi, true, false);
        if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wo, wi);
        if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wo, wi, spec_albedo, alpha, true);
        weight  = diff_albedo*diff_f + spec_f;

        pdf *= detach(pSpecularReflection);
        float test_roughness = sqrt(alpha);
        if (test_roughness > 0.15f )
        {
            if (pDiffuseReflection > 0.f) pdf += detach(pDiffuseReflection) * DiffuseReflection_evalPdf(wo, wi);
        }
        else
        {
            specularBounce = 1;
        }

        weight /= pdf;
        diff_f /= pdf;
        spec_f /= pdf;
    }

    diff_weight = diff_f;
    spec_weight = spec_f;
    
    return valid;
}

[PreferRecompute]
bool FalcorBRDF_sample_no_weight(float pDiffuseReflection, float pSpecularReflection,
                        float3 wo, out float3 wi, no_diff out float pdf, no_diff inout uint specularBounce, no_diff inout uint sg,
    float alpha, float3 spec_albedo, float3 diff_albedo, bool activeLobes = false, bool allowDeltaEval = false)
{
    wi = {};
    pdf = 0.f;
    specularBounce = 0;
    float3 weight = float3(0.f);

    //float3 diff_f = float3(0.0, 0.0, 0.0);
    //float3 spec_f = float3(0.0, 0.0, 0.0);

    bool valid = false;
    float uSelect = sampleNext1D(sg);

    if (uSelect < pDiffuseReflection)
    {
        valid = DiffuseReflection_sample(diff_albedo, wo, wi, pdf, weight, sg);

        //weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
        //                         wo, wi, true, false);
        //if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wo, wi);
        //if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wo, wi, spec_albedo, alpha, true);
        //weight  = diff_albedo*diff_f + spec_f;
                        
        pdf *= detach(pDiffuseReflection);
        if (pSpecularReflection > 0.f) pdf += detach(pSpecularReflection) * SpecularReflection_evalPdf(wo, wi, alpha, activeLobes, allowDeltaEval);

        //weight /= pdf;
        //diff_f /= pdf;
        //spec_f /= pdf;
    }
    else if (uSelect < pDiffuseReflection + pSpecularReflection)
    {
        valid = SpecularReflection_sample(spec_albedo, alpha, wo, wi, pdf, weight, sg, activeLobes, allowDeltaEval);
        //weight = FalcorBRDF_eval(pDiffuseReflection, pSpecularReflection, alpha, spec_albedo, diff_albedo,
        //                         wo, wi, true, false);
        //if (pDiffuseReflection > 0.f) diff_f = Diffuse_light(wo, wi);
        //if (pSpecularReflection > 0.f) spec_f = SpecularReflection_eval(wo, wi, spec_albedo, alpha, true);
        //weight  = diff_albedo*diff_f + spec_f;

        pdf *= detach(pSpecularReflection);
        float test_roughness = sqrt(alpha);
        if (test_roughness > 0.15f )
        {
            if (pDiffuseReflection > 0.f) pdf += detach(pDiffuseReflection) * DiffuseReflection_evalPdf(wo, wi);
        }
        else
        {
            specularBounce = 1;
        }

        //weight /= pdf;
        //diff_f /= pdf;
        //spec_f /= pdf;
    }

    //diff_weight = diff_f;
    //spec_weight = spec_f;
    
    return valid;
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