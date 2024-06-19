import utils.light;
import utils.random;
import utils.helper;

static const uint kLightTileCount = LIGHT_TILE_COUNT;
static const uint kLightTileSize = LIGHT_TILE_SIZE;

[PreferRecompute]
float getStratifiedRandom(uint sampleIndex, uint sampleCount, float offset)
{
    return frac(offset + float(sampleIndex) / float(sampleCount));
}

[AutoPyBindCUDA]
[CUDAKernel]
void process_GenerateLightTiles(TensorView<float> env_tex, 
    TensorView<float> pdf_, TensorView<float> cdf_, TensorView<float> mpdf_, TensorView<float> mcdf_,
    uint width, uint height, uint frameIndex,
    TensorView<float> light_data, TensorView<int> light_uv, TensorView<float> light_inv_pdf,
    TensorView<float> debug_out)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint tileIndex = dispatchIdx.y;
    uint sampleIndex = dispatchIdx.x;
    if (tileIndex >= kLightTileCount || sampleIndex >= kLightTileSize) return;

    uint bufferIndex = tileIndex * kLightTileSize + sampleIndex;
    
    // Setup per-tile random generator and pick random offset.
    uint tileSg = Seed_Generator(tileIndex, frameIndex);
    float offset = sampleNext1D(tileSg);
    // Setup per-thread random generator.
    uint sg = Seed_Generator(bufferIndex, frameIndex + 1);

    uint kEnvLightSampleCount = kLightTileSize;

    PresampledLight lightSample = createInvalid_PresampledLight();

    //environment map
    float random = getStratifiedRandom(sampleIndex, kEnvLightSampleCount, offset);
    lightSample = presampled_light(random, sg, env_tex, width, height, pdf_, cdf_, mpdf_, mcdf_);

    // write light sample.
    
    light_data[bufferIndex,0] = lightSample.light_data.x;
    light_data[bufferIndex,1] = lightSample.light_data.y;
    light_data[bufferIndex,2] = lightSample.light_data.z;
    light_uv[bufferIndex,0] = lightSample.light_uv.x;
    light_uv[bufferIndex,1] = lightSample.light_uv.y;
    light_inv_pdf[bufferIndex] = lightSample.inv_pdf;

    //debug space
    /*if(lightSample.light_data.x > 0.1f)
    {
        int tex_x = lightSample.light_uv.x;
        int tex_y = lightSample.light_uv.y;
        debug_out[get_data_idx(tex_x, tex_y, width),0] = env_tex[get_data_idx(tex_x, tex_y, width), 0];
        debug_out[get_data_idx(tex_x, tex_y, width),1] = env_tex[get_data_idx(tex_x, tex_y, width), 1];
        debug_out[get_data_idx(tex_x, tex_y, width),2] = env_tex[get_data_idx(tex_x, tex_y, width), 2];
    }*/
}