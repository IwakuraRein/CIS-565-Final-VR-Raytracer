/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

 //-------------------------------------------------------------------------------------------------
 // This file is the main function for the path tracer.
 // * `samplePixel()` is setting a ray from the camera origin through a pixel (jitter)
 // * `IndirectSample()` will loop until the ray depth is reached or the environment is hit.
 // * `DirectLight()` is the contribution at the hit, if the shadow ray is not hitting anything.

#define ENVMAP 1
#define RR 1        // Using russian roulette
#define RR_DEPTH 0  // Minimum depth

#include "pbr_metallicworkflow.glsl"
#include "gltf_material.glsl"
#include "punctual.glsl"
#include "env_sampling.glsl"
#include "shade_state.glsl"
#include "reservoir.glsl"

float dummyPdf;

bool IsPdfInvalid(float p) {
    return p <= 1e-8 || isnan(p);
}

bool Occlusion(Ray ray, State state, float dist) {
    return AnyHit(ray, dist - abs(ray.origin.x - state.position.x) -
        abs(ray.origin.y - state.position.y) -
        abs(ray.origin.z - state.position.z));
}

vec3 BSDF(State state, vec3 V, vec3 N, vec3 L) {
    return metallicWorkflowBSDF(state, N, V, L);
}

float Pdf(State state, vec3 V, vec3 N, vec3 L) {
    return metallicWorkflowPdf(state, N, V, L);
}

vec3 Eval(State state, vec3 V, vec3 N, vec3 L, inout float pdf) {
     return metallicWorkflowEval(state, N, V, L, pdf);
}

vec3 Sample(State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed) {
     return metallicWorkflowSample(state, N, V, vec3(rand(seed), rand(seed), rand(seed)), L, pdf);
}

vec3 EnvRadiance(vec3 dir) {
    if (_sunAndSky.in_use == 1)
        return sun_and_sky(_sunAndSky, dir) * rtxState.hdrMultiplier;
    else {
        vec2 uv = GetSphericalUv(dir);
        return texture(environmentTexture, uv).rgb * rtxState.hdrMultiplier;
    }
}

float EnvPdf(vec3 dir) {
    float pdf;
    if (_sunAndSky.in_use == 1) {
        pdf = 0.5;
    }
    else {
        vec2 uv = GetSphericalUv(dir);
        pdf = luminance(texture(environmentTexture, uv).rgb) * rtxState.envMapLuminIntegInv;
    }
    return pdf * rtxState.environmentProb;
}

vec3 EnvEval(vec3 dir, out float pdf) {
    if (_sunAndSky.in_use == 1) {
        pdf = 0.5 * rtxState.environmentProb;
        return sun_and_sky(_sunAndSky, dir) * rtxState.hdrMultiplier;
    }
    else {
        vec2 uv = GetSphericalUv(dir);
        vec3 radiance = texture(environmentTexture, uv).rgb;
        pdf = luminance(radiance) * rtxState.envMapLuminIntegInv * rtxState.environmentProb;
        return radiance;
    }
}

vec3 LightEval(State state, float dist, vec3 dir, out float pdf) {
    float lightProb = (1.0 - rtxState.environmentProb);

    GltfShadeMaterial mat = materials[state.matID];
    vec3 emission = mat.emissiveFactor;

    pdf = luminance(emission) * rtxState.lightLuminIntegInv * lightProb;
    pdf *= dist * dist / absDot(state.ffnormal, dir);

    if (mat.emissiveTexture > -1) {
        vec2 uv = state.texCoord;
        emission *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(mat.emissiveTexture)], uv, 0)).rgb;
    }
    return emission / state.area;
}

vec2 SampleTriangleUniform(vec3 v0, vec3 v1, vec3 v2) {
    float ru = rand(prd.seed);
    float rv = rand(prd.seed);
    float r = sqrt(rv);
    float u = 1.0 - r;
    float v = ru * r;
    return vec2(u, v);
}

float TriangleArea(vec3 v0, vec3 v1, vec3 v2) {
    return length(cross(v1 - v0, v2 - v0)) * 0.5;
}

float SampleTriangleLight(vec3 x, out LightSample lightSample) {
    if (lightBufInfo.trigLightSize == 0) {
        return InvalidPdf;
    }

    int id = min(int(float(lightBufInfo.trigLightSize) * rand(prd.seed)), int(lightBufInfo.trigLightSize) - 1);

    if (rand(prd.seed) > trigLights[id].impSamp.q) {
        id = trigLights[id].impSamp.alias;
    }

    TrigLight light = trigLights[id];

    vec3 v0 = light.v0;
    vec3 v1 = light.v1;
    vec3 v2 = light.v2;

    vec3 normal = cross(v1 - v0, v2 - v0);
    float area = length(normal) * 0.5;
    normal = normalize(normal);

    vec2 baryCoord = SampleTriangleUniform(v0, v1, v2);
    vec3 y = baryCoord.x * v0 + baryCoord.y * v1 + (1 - baryCoord.x - baryCoord.y) * v2;

    GltfShadeMaterial mat = materials[light.matIndex];
    vec3 emission = mat.emissiveFactor;
    if (mat.emissiveTexture > -1) {
        vec2 uv = baryCoord.x * light.uv0 + baryCoord.y * light.uv1 + (1 - baryCoord.x - baryCoord.y) * light.uv2;
        emission *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(mat.emissiveTexture)], uv, 0)).rgb;
    }
    vec3 dir = y - x;
    float dist = length(dir);
    lightSample.Li = emission / area;
    lightSample.wi = dir / dist;
    lightSample.dist = dist;
    return light.impSamp.pdf * (dist * dist) / (area * abs(dot(lightSample.wi, normal)));
}

float SamplePuncLight(vec3 x, out LightSample lightSample) {
    if (lightBufInfo.puncLightSize == 0) {
        return InvalidPdf;
    }

    int id = min(int(float(lightBufInfo.trigLightSize) * rand(prd.seed)), int(lightBufInfo.puncLightSize) - 1);

    if (rand(prd.seed) > puncLights[id].impSamp.q) {
        id = puncLights[id].impSamp.alias;
    }

    PuncLight light = puncLights[id];
    vec3 dir = light.position - x;
    float dist = length(dir);
    lightSample.Li = light.color * light.intensity / (dist * dist);
    lightSample.wi = dir / dist;
    lightSample.dist = dist;
    return light.impSamp.pdf;
}

float SampleDirectLightNoVisibility(vec3 pos, out LightSample lightSample) {
    float rnd = rand(prd.seed);
    if (rnd < rtxState.environmentProb) {
        // Sample environment
        vec4 dirAndPdf = EnvSample(lightSample.Li);
        if (IsPdfInvalid(dirAndPdf.w)) {
            return InvalidPdf;
        }
        lightSample.wi = dirAndPdf.xyz;
        lightSample.dist = INFINITY;
        return dirAndPdf.w * rtxState.environmentProb;
    }
    else {
        if (rnd < rtxState.environmentProb + (1.0 - rtxState.environmentProb) * lightBufInfo.trigSampProb) {
            // Sample triangle mesh light
            return (1.0 - rtxState.environmentProb) * SampleTriangleLight(pos, lightSample) * lightBufInfo.trigSampProb;
        }
        else {
            // Sample point light
            return (1.0 - rtxState.environmentProb) * SamplePuncLight(pos, lightSample) * (1.0 - lightBufInfo.trigSampProb);
        }
    }
}

float SampleDirectLight(State state, out vec3 radiance, out vec3 dir) {
    LightSample lsample;
    float pdf = SampleDirectLightNoVisibility(state.position, lsample);
    if (IsPdfInvalid(pdf)) {
        return InvalidPdf;
    }

    Ray shadowRay;
    shadowRay.origin = OffsetRay(state.position, state.ffnormal);
    shadowRay.direction = lsample.wi;

    if (Occlusion(shadowRay, state, lsample.dist)) {
        return InvalidPdf;
    }
    radiance = lsample.Li;
    dir = lsample.wi;
    return pdf;
}

vec3 DirectLight(State state, vec3 wo) { // importance sample on light sources
    LightSample lightSample;
    float pdf = SampleDirectLightNoVisibility(state.position, lightSample);
    if (IsPdfInvalid(pdf)) {
        return vec3(0.0);
    }

    Ray shadowRay;
    shadowRay.origin = OffsetRay(state.position, state.ffnormal);
    shadowRay.direction = lightSample.wi;

    if (Occlusion(shadowRay, state, lightSample.dist)) {
        return vec3(0.0);
    }
    return lightSample.Li * Eval(state, wo, state.ffnormal, lightSample.wi, dummyPdf) *
        max(dot(state.ffnormal, lightSample.wi), 0.0) / pdf;
}

/*
bool UpdateSample(inout Ray r, in State state, in float screenDepth, inout vec3 radiance, inout vec3 throughput) {
    // Emissive material
    radiance += state.mat.emission * throughput;
    // add screenspace indirect
    vec3 ndc = vec3(sceneCamera.projView * vec4(state.position, 1.0));
    ndc += vec3(1, 1, 0);
    ndc *= 0.5;
    //check if this sample appears in direct stage's result
    if (ndc.x >= 0.0 && ndc.x <= 1.0 && ndc.y >= 0.0 && ndc.y <= 1.0) {
        // TODO: adjust offset according to normal
        if (length(state.position - sceneCamera.lastPosition) <= screenDepth * 1.01) {
            ivec2 coord = ivec2(round(ndc.x * rtxState.size.x), round(ndc.y * rtxState.size.y));
            LightSample cache = thisDirectResv[coord.y * rtxState.size.x + coord.x].lightSample;

            cache.Li *= Eval(state, -r.direction, state.ffnormal, cache.wi, dummyPdf);
            if (dummyPdf > 0)
                radiance += cache.Li * throughput;
        }
    }

    // add direct light
    float directProb = min(state.mat.roughness * 2.0, 1.0);
    if (rand(prd.seed) < directProb) { // importance sampling on light sources
        radiance += DirectLight(state, -r.direction) * throughput / directProb;
    }

    BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
    bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
    bsdfSampleRec.L = normalize(bsdfSampleRec.L);

    if (bsdfSampleRec.pdf > 0.0) {
        throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
    }
    else {
        return false;
    }

    // Next ray
    r.direction = bsdfSampleRec.L;
    r.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

    return !state.isEmitter;
}
bool UpdateSampleWithoutEmission(inout Ray r, in State state, inout vec3 radiance, inout vec3 throughput) {
    // KHR_materials_unlit
    if (state.isEmitter) {
        return false;
    }
    // add direct light
    float directProb = min(state.mat.roughness * 2.0, 1.0);
    if (rand(prd.seed) < directProb) { // importance sampling on light sources
        radiance += DirectLight(state, -r.direction) * throughput / directProb;
    }

    BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
    bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
    bsdfSampleRec.L = normalize(bsdfSampleRec.L);

    if (bsdfSampleRec.pdf > 0.0) {
        throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
    }
    else {
        return false;
    }

    // Next ray
    r.direction = bsdfSampleRec.L;
    r.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

    return true;
}
bool UpdateSampleWithoutEmissionDirectLight(inout Ray r, in State state, inout vec3 radiance, inout vec3 throughput) {
    if (state.isEmitter) {
        return false;
    }

    BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
    bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
    bsdfSampleRec.L = normalize(bsdfSampleRec.L);

    if (bsdfSampleRec.pdf > 0.0) {
        throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
    }
    else {
        return false;
    }

    // Next ray
    r.direction = bsdfSampleRec.L;
    r.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

    return true;
}

vec3 IndirectSample(Ray r, State state, float hitT) {
    prd.hitT = hitT;
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    { // first intersection
      // we don't want to reintroduce the luminance that direct stage has already done
        if (!UpdateSampleWithoutEmissionDirectLight(r, state, radiance, throughput))
            return radiance;
#ifdef RR
        // For Russian-Roulette (minimizing live state)
        float rrPcont = (0 >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
        if (rand(prd.seed) >= rrPcont)
            return radiance;                // paths with low throughput that won't contribute
        throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
    }

    { // second intersection
        ClosestHit(r);

        // Hitting the environment
        if (prd.hitT >= INFINITY) {
            return radiance;
        }

        // Get Position, Normal,Texture Coordinates
        state = GetState(prd, r.direction);

        // Filling material structures
        GetMaterials(state, r);

        if (!UpdateSampleWithoutEmission(r, state, radiance, throughput))
            return radiance;

#ifdef RR
        // For Russian-Roulette (minimizing live state)
        float rrPcont = (1 >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
        if (rand(prd.seed) >= rrPcont)
            return radiance;                // paths with low throughput that won't contribute
        throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
    }

    { // second bounce and more
        for (int depth = 2; depth < rtxState.maxDepth; depth++) {
            ClosestHit(r);

            // Hitting the environment
            if (prd.hitT >= INFINITY) {
                vec3 env;
                if (_sunAndSky.in_use == 1)
                    env = sun_and_sky(_sunAndSky, r.direction);
                else {
                    vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
                    env = texture(environmentTexture, uv).rgb;
                }
                // Done sampling return
                return radiance + (env * rtxState.hdrMultiplier * throughput);
            }

            // Get Position, Normal, Texture Coordinates
            state = GetState(prd, r.direction);

            // Filling material structures
            GetMaterials(state, r);

            if (!UpdateSample(r, state, hitT, radiance, throughput))
                return radiance;

#ifdef RR
            // For Russian-Roulette (minimizing live state)
            float rrPcont = (depth >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
            if (rand(prd.seed) >= rrPcont)
                return radiance;                // paths with low throughput that won't contribute
            throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
        }
    }

    return radiance;
}
*/

vec3 clampRadiance(vec3 radiance) {
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)) {
        return vec3(0.0);
    }

    float lum = luminance(radiance);
    if (lum > rtxState.fireflyClampThreshold) {
        radiance *= rtxState.fireflyClampThreshold / lum;
    }
    return radiance;
}

void loadLastGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth) {
    uvec2 gInfo = imageLoad(lastGbuffer, imageCoords).xy;
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
}

void loadLastGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth, out uint matHash) {
    uvec4 gInfo = imageLoad(lastGbuffer, imageCoords);
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
    matHash = gInfo.w & 0xFF000000;
}

void loadThisGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth) {
    uvec2 gInfo = imageLoad(thisGbuffer, imageCoords).xy;
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
}

void loadThisGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth, out uint matHash) {
    uvec4 gInfo = imageLoad(thisGbuffer, imageCoords);
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
    matHash = gInfo.w & 0xFF000000;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
Ray raySpawn(ivec2 imageCoords, ivec2 sizeImage) {
    // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    // vec2 subpixel_jitter = rtxState.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(prd.seed), rand(prd.seed));

    // Compute sampling position between [-1 .. 1]
    const vec2 pixelCenter = vec2(imageCoords) + 0.5;
    const vec2 inUV = pixelCenter / vec2(sizeImage.xy);
    vec2 d = inUV * 2.0 - 1.0;

    // Compute ray origin and direction
    vec4 origin = sceneCamera.viewInverse * vec4(0, 0, 0, 1);
    vec4 target = sceneCamera.projInverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = sceneCamera.viewInverse * vec4(normalize(target.xyz), 0);

    // Depth-of-Field
    // vec3 focalPoint = sceneCamera.focalDist * direction.xyz;
    // float cam_r1 = rand(prd.seed) * M_TWO_PI;
    //float cam_r2 = rand(prd.seed) * sceneCamera.aperture;
    // float cam_r2 = 0.0;
    // vec4 cam_right = sceneCamera.viewInverse * vec4(1, 0, 0, 0);
    // vec4 cam_up = sceneCamera.viewInverse * vec4(0, 1, 0, 0);
    // vec3 randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
    // vec3 finalRayDir = normalize(focalPoint - randomAperturePos);

    return Ray(origin.xyz, /*finalRayDir*/ direction.xyz);
}

vec3 DebugInfo(in State state) {
    switch (rtxState.debugging_mode) {
    case eMetallic:
        return vec3(state.mat.metallic);
    case eNormal:
        return (state.normal + vec3(1)) * .5;
    case eDepth:
        return vec3(0.0);
    case eBaseColor:
        return state.mat.albedo;
    case eEmissive:
        return state.mat.emission;
    case eRoughness:
        return vec3(state.mat.roughness);
    case eTexcoord:
        return vec3(state.texCoord, 0);
    };
    return vec3(1000, 0, 0);
}