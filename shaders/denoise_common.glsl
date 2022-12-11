#ifndef DENOISE_COMMON_GLSL
#define DENOISE_COMMON_GLSL

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

vec3 getCameraPos(ivec2 coord, float dist, ivec2 imageSize) {
    Ray ray = raySpawn(coord, imageSize);
    return ray.origin + ray.direction * dist;
}

void loadThisGeometry(ivec2 coord, out vec3 normal, out vec3 pos, out uint matHash, ivec2 imageSize) {
    uvec4 gInfo = imageLoad(thisGbuffer, coord);
    normal = decompress_unit_vec(gInfo.y);
    pos = getCameraPos(coord, uintBitsToFloat(gInfo.x), imageSize);
    matHash = gInfo.w & 0xFF000000;
}

void loadThisGeometry(ivec2 coord, out vec3 albedo, out vec3 normal, out vec3 pos, out uint matHash, ivec2 imageSize) {
    uvec4 gInfo = imageLoad(thisGbuffer, coord);
    albedo = unpackUnorm4x8(gInfo.w).rgb;
    normal = decompress_unit_vec(gInfo.y);
    pos = getCameraPos(coord, uintBitsToFloat(gInfo.x), imageSize);
    matHash = gInfo.w & 0xFF000000;
}

#endif