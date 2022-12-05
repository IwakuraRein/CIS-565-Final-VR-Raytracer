#include "host_device.h"
#include "globals.glsl"
#include "layouts.glsl"
#include "random.glsl"
#include "common.glsl"
#include "compress.glsl"

layout(push_constant) uniform _RtxState{
    RtxState rtxState;
};

const float Gaussian5x5[5][5] = {
	{ .0030f, .0133f, .0219f, .0133f, .0030f },
	{ .0133f, .0596f, .0983f, .0596f, .0133f },
	{ .0219f, .0983f, .1621f, .0983f, .0219f },
	{ .0133f, .0596f, .0983f, .0596f, .0133f },
	{ .0030f, .0133f, .0219f, .0133f, .0030f }
};

float luminance(vec3 rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

Ray raySpawn(ivec2 coord, ivec2 sizeImage) {
    const vec2 pixelCenter = vec2(coord) + 0.5;
    const vec2 inUV = pixelCenter / vec2(sizeImage.xy);
    vec2 d = inUV * 2.0 - 1.0;
    vec4 origin = sceneCamera.viewInverse * vec4(0, 0, 0, 1);
    vec4 target = sceneCamera.projInverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = sceneCamera.viewInverse * vec4(normalize(target.xyz), 0);
    return Ray(origin.xyz, direction.xyz);
}

vec3 getCameraPos(ivec2 coord, float dist) {
    Ray ray = raySpawn(coord, rtxState.size);
    return ray.origin + ray.direction * dist;
}

void loadThisGeometry(ivec2 coord, out vec3 normal, out vec3 pos, out uint matHash) {
    uvec4 gInfo = imageLoad(thisGbuffer, coord);
    normal = decompress_unit_vec(gInfo.y);
    pos = getCameraPos(coord, uintBitsToFloat(gInfo.x));
    matHash = gInfo.w & 0xFF000000;
}

void loadThisGeometry(ivec2 coord, out vec3 albedo, out vec3 normal, out vec3 pos, out uint matHash) {
    uvec4 gInfo = imageLoad(thisGbuffer, coord);
    albedo = unpackUnorm4x8(gInfo.w).rgb;
    normal = decompress_unit_vec(gInfo.y);
    pos = getCameraPos(coord, uintBitsToFloat(gInfo.x));
    matHash = gInfo.w & 0xFF000000;
}

vec3 waveletFilter(image2D inImage, ivec2 coord, vec3 norm, vec3 pos, uint matHash,
    float sigLumin, float sigNormal, float sigDepth, int level
) {
    if (matHash == InvalidMatId) {
        return vec3(0.0);
    }
    int step = 1 << level;

    vec3 sum = vec3(0.0);
    float sumWeight = 0.0;

    vec3 color = imageLoad(inImage, coord).rgb;

    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            ivec2 q = coord + ivec2(i, j) * step;
            if (q.x >= rtxState.size.x || q.y >= rtxState.size.y ||
                q.x < 0 || q.y < 0) {
                continue;
            }

            vec3 normQ; vec3 posQ; uint matHashQ;
            loadThisGeometry(q, normQ, posQ, matHashQ);
            vec3 colorQ = imageLoad(inImage, q).rgb;

            if (matHash != matHashQ || matHashQ == InvalidMatId) {
                continue;
            }

            float var = sigLumin;
            float distColor = abs(luminance(color) - luminance(colorQ));
            float wColor = exp(-distColor / var) + 1e-3;
            //float distColor = dot(color - colorQ, color - colorQ);
            //float wColor = exp(-distColor / rtxState.sigLumin) + 1e-3;

            float distNorm2 = dot(norm - normQ, norm - normQ);
            float wNorm = min(1.0, exp(-distNorm2 / sigNormal));

            float distPos2 = dot(pos - posQ, pos - posQ);
            float wDepth = exp(-distPos2 / sigDepth) + 1e-3;

            float weight = wColor * wNorm * wDepth * Gaussian5x5[i + 2][j + 2];
            sum += colorQ * weight;
            sumWeight += weight;
        }
    }
    return (sumWeight < 1e-6) ? color : sum / sumWeight;
}