[PreferRecompute]
uint interleave_32bit(uint2 v)
{
    uint x = v.x & 0x0000ffff;              // x = ---- ---- ---- ---- fedc ba98 7654 3210
    uint y = v.y & 0x0000ffff;

    x = (x | (x << 8)) & 0x00FF00FF;        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x | (x << 4)) & 0x0F0F0F0F;        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x | (x << 2)) & 0x33333333;        // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x | (x << 1)) & 0x55555555;        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}
[PreferRecompute]
uint2 blockCipherTEA(uint v0, uint v1, uint iterations = 16)
{
    uint sum = 0;
    const uint delta = 0x9e3779b9;
    const uint k[4] = { 0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e }; // 128-bit key.
    for (uint i = 0; i < iterations; i++)
    {
        sum += delta;
        v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }
    return uint2(v0, v1);
}
[PreferRecompute]
uint Seed_Generator(uint2 pixel, uint sampleNumber)
{
    // Use block cipher to generate a pseudorandom initial seed.
    uint seed = blockCipherTEA(interleave_32bit(pixel), sampleNumber).x;
    return seed;
}
[PreferRecompute]
uint nextRandom(inout uint state)
{
    const uint A = 1664525u;
    const uint C = 1013904223u;
    state = (A * state + C);
    return state;
}
[PreferRecompute]
float sampleNext1D(inout uint seed)
{
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    uint bits = nextRandom(seed);
    return (bits >> 8) * 0x1p-24;
}
[PreferRecompute]
float2 sampleNext2D(inout uint sg)
{
    float2 sample;
    // Don't use the float2 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    return sample;
}
[PreferRecompute]
float3 sampleNext3D(inout uint sg)
{
    float3 sample;
    // Don't use the float3 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    sample.z = sampleNext1D(sg);
    return sample;
}