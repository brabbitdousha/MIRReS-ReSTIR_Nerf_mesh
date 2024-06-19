import brdf;
import helperDi;
import lightDi;
import random;
struct Reservoir
{
    TensorView<float> light_data;
    TensorView<float> light_pdf;//< Selected light sample.
    TensorView<int> M; //< Number of samples seen so far.
    TensorView<float> weight; //< Probabilistic weight.
};
struct reservoir
{
    float3 light_data;
    float light_pdf;//< Selected light sample.

    int M;                    ///< Number of samples seen so far (float required for pairwise MIS).
    float weight;               ///< Probabilistic weight.
}
struct RisState
{
    float3 light_data;
    int2 light_uv;
    float inv_pdf;//< Selected light sample.

    float weightSum;            ///< Sum of weights seen so far.
    float M;                    ///< Number of samples seen so far (float required for pairwise MIS).
    float weight;               ///< Probabilistic weight.
    float canonicalWeight;      ///< Canonical sample weight factor (required for pairwise MIS).
}
struct FinalSample
{
    TensorView<float> dir;
    TensorView<float> distance;
    DiffTensorView<float> Li;
}
[PreferRecompute]
RisState createEmpty_RisState()
{
    RisState ls;
    ls.light_data = float3(0.f);
    ls.light_uv = int2(0,0);
    ls.inv_pdf = 0.f;

    ls.weightSum = 0.f;
    ls.M = 0.f;
    ls.weight = 0.f;
    ls.canonicalWeight = 0.f; 

    return ls;
}

float mFactor(float q0, float q1)
{
    return q0 == 0.f ? 1.f : clamp(pow(min(q1 / q0, 1.f), 8.f), 0.f, 1.f);
}

float pairwiseMisWeight(float q0, float q1, float N0, float N1)
{
    return (q1 == 0.f) ? 0.f : (N0 * q0) / (q0 * N0 + q1 * N1);
}

bool isValidNeighbor(float3 currNormal, float currDepth, float3 otherNormal, float otherDepth, float normalThreshold, float depthThreshold)
{
    return
        dot(currNormal, otherNormal) >= normalThreshold &&
        abs((currDepth - otherDepth)) <= depthThreshold * currDepth;
}

float evalTargetFunction(float3 light_emission, float3 light_dir, float3 normal, float3 ray_dir, float3 brdf,
    float curr_vis, const bool withVisibility = false)
{
    float brdfWeight = evalBRDF(light_dir, -ray_dir, normal, brdf.z, brdf.x, brdf.y);
    float weight = max(0.f, luminance(light_emission) * brdfWeight);
    if (withVisibility) weight *= curr_vis;
    return weight;
}

float evalInitialSamplePdf(bool kSpecularOnly, float kInitialBRDFSampleCount, float kBRDFSampleRatio, 
float3 light_dir, float light_pdf, float3 V, float3 N, float ggxAlpha, float diffuseWeight, float specularWeight)
{
    if (kInitialBRDFSampleCount > 0)
    {
        float brdfPdf = evalPdfBRDF(kSpecularOnly, light_dir, V, N, ggxAlpha, diffuseWeight, specularWeight);
        return lerp(light_pdf, brdfPdf, kBRDFSampleRatio);
    }
    else
    {
        return light_pdf;
    }
}

bool streamingResampleStep(inout RisState state, const PresampledLight lightSample, float targetPdf, float sourcePdf, inout uint sg)
{
    float sampleWeight = targetPdf / sourcePdf;

    state.weightSum += sampleWeight;
    state.M += 1.f;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.light_data = lightSample.light_data;
        state.light_uv = lightSample.light_uv;
        state.inv_pdf = lightSample.inv_pdf;//light sample

        state.weight = targetPdf;
    }

    // this return value does not matter since it is not used anywhere
    return selectSample;
}

// used by resampling
bool streamingResampleStep(inout RisState state, const reservoir reservoir_, float targetPdf, inout uint sg, bool isCenterPixel)
{
    float sampleWeight = targetPdf * reservoir_.weight * reservoir_.M;

    state.weightSum += sampleWeight;
    state.M += reservoir_.M;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.light_data = reservoir_.light_data;
        state.inv_pdf = reservoir_.light_pdf;//light sample

        state.weight = targetPdf;
    }

    return selectSample;
}

bool streamingResampleStepMis(
    inout RisState state,
    const reservoir canonicalReservoir, float3 canonical_light_emission, float3 canonical_light_dir, 
    float canonicalTargetPdf, const float3 curr_normal, const float3 curr_ray_dir, const float3 curr_brdf,
    const reservoir candidateReservoir, float3 candidate_light_emission, float3 candidate_light_dir,
    const float3 candidate_normal, const float3 candidate_ray_dir, const float3 candidate_brdf,
    uint k,
    inout uint sg)
{
    float candidateTargetPdf = evalTargetFunction(candidate_light_emission, candidate_light_dir, 
        candidate_normal, candidate_ray_dir, candidate_brdf, 1.0f);
    float candidateTargetPdfAtOther = evalTargetFunction(candidate_light_emission, candidate_light_dir, 
        curr_normal, curr_ray_dir, curr_brdf, 1.0f);
    float canonicalTargetPdfAtOther = evalTargetFunction(canonical_light_emission, canonical_light_dir, 
        candidate_normal, candidate_ray_dir, candidate_brdf, 1.0f);
    
    float m0 =       pairwiseMisWeight(candidateTargetPdf,        candidateTargetPdfAtOther, candidateReservoir.M * k, canonicalReservoir.M);
    float m1 = 1.f - pairwiseMisWeight(canonicalTargetPdfAtOther, canonicalTargetPdf,        candidateReservoir.M * k, canonicalReservoir.M);
    
    float sampleWeight = candidateTargetPdfAtOther * candidateReservoir.weight * m0;
    state.M += candidateReservoir.M * min(mFactor(candidateTargetPdf, candidateTargetPdfAtOther), mFactor(canonicalTargetPdfAtOther, canonicalTargetPdf));
    state.weightSum += sampleWeight;
    state.canonicalWeight += m1;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.light_data = candidateReservoir.light_data;
        state.inv_pdf = candidateReservoir.light_pdf;//light sample

        state.weight = candidateTargetPdfAtOther;
    }

    return selectSample;
}

bool streamingResampleStepMisUnbiased(
    inout RisState state,
    const reservoir canonicalReservoir, float3 canonical_light_emission, float3 canonical_light_dir, 
    float canonicalTargetPdf, const float3 curr_normal, const float3 curr_ray_dir, const float3 curr_brdf,
    const reservoir candidateReservoir, float3 candidate_light_emission, float3 candidate_light_dir,
    const float3 candidate_normal, const float3 candidate_ray_dir, const float3 candidate_brdf,
    float canonicalVisibility, float candidateVisibility,
    uint k,
    inout uint sg)
{
    float candidateTargetPdf = evalTargetFunction(candidate_light_emission, candidate_light_dir, 
        candidate_normal, candidate_ray_dir, candidate_brdf, 1.0f);
    float candidateTargetPdfAtOther = evalTargetFunction(candidate_light_emission, candidate_light_dir, 
        curr_normal, curr_ray_dir, curr_brdf, 1.0f);
    float canonicalTargetPdfAtOther = evalTargetFunction(canonical_light_emission, canonical_light_dir, 
        candidate_normal, candidate_ray_dir, candidate_brdf, 1.0f);
    
    candidateTargetPdfAtOther *= canonicalVisibility;
    canonicalTargetPdfAtOther *= candidateVisibility;

    float m0 = pairwiseMisWeight(candidateTargetPdf, candidateTargetPdfAtOther, candidateReservoir.M * k, canonicalReservoir.M);
    float m1 = 1.f - pairwiseMisWeight(canonicalTargetPdfAtOther, canonicalTargetPdf, candidateReservoir.M * k, canonicalReservoir.M);

    float sampleWeight = candidateTargetPdfAtOther * candidateReservoir.weight * m0;
    state.M += candidateReservoir.M * min(mFactor(candidateTargetPdf, candidateTargetPdfAtOther), mFactor(canonicalTargetPdfAtOther, canonicalTargetPdf));
    state.weightSum += sampleWeight;
    state.canonicalWeight += m1;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.light_data = candidateReservoir.light_data;
        state.inv_pdf = candidateReservoir.light_pdf;//light sample

        state.weight = candidateTargetPdfAtOther;
    }

    return selectSample;
}

bool streamingResampleFinalizeMis(inout RisState state, const reservoir canonicalReservoir, float canonicalTargetPdf, inout uint sg)
{
    float sampleWeight = canonicalTargetPdf * canonicalReservoir.weight * state.canonicalWeight;

    state.M += canonicalReservoir.M;
    state.weightSum += sampleWeight;

    bool selectSample = sampleNext1D(sg) * state.weightSum < sampleWeight;

    if (selectSample)
    {
        state.light_data = canonicalReservoir.light_data;
        state.inv_pdf = canonicalReservoir.light_pdf;//light sample

        state.weight = canonicalTargetPdf;
    }

    return selectSample;
}