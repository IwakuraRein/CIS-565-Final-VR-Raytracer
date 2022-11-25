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
#include "restir.glsl"

float dummyPdf;

bool IsPdfInvalid(float p) {
    return p <= 0.0 || isnan(p);
}

State GetState(Ray incomingRay) {
    State state;
    ShadeState sstate = GetShadeState(prd);
    state.position = sstate.position;
    state.normal = sstate.normal;
    state.tangent = sstate.tangent_u[0];
    state.bitangent = sstate.tangent_v[0];
    state.texCoord = sstate.text_coords[0];
    state.matID = sstate.matIndex;
    state.ffnormal = dot(state.normal, incomingRay.direction) <= 0.0 ? state.normal : -state.normal;
    return state;
}

bool Occlusion(Ray ray, State state, float dist) {
    return AnyHit(ray, dist - abs(ray.origin.x - state.position.x) -
        abs(ray.origin.y - state.position.y) -
        abs(ray.origin.z - state.position.z));
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 Eval(in State state, in vec3 V, in vec3 N, in vec3 L, inout float pdf) {
    // switch (rtxState.pbrMode) {
    // case 0:
    //     return DisneyEval(state, V, N, L, pdf);
    // case 1:
    //     return PbrEval(state, V, N, L, pdf);
    // case 2:
        return metallicWorkflowEval(state, N, V, L, pdf);
    // }
    // return vec3(0.0);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 Sample(in State state, in vec3 V, in vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed) {
    // switch (rtxState.pbrMode) {
    // case 0:
    //     return DisneySample(state, V, N, L, pdf, seed);
    // case 1:
    //     return PbrSample(state, V, N, L, pdf, seed);
    // case 2:
        return metallicWorkflowSample(state, N, V, vec3(rand(seed), rand(seed), rand(seed)), L, pdf);
    // }
    // return vec3(0.0);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
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

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
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

/*
* Sample direct light sources without occlusion test
* Return float: pdf
*/
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
        if (getDepth(ndc.z) <= (screenDepth + 1e-3)) {
            ivec2 coord = ivec2(round(ndc.x * rtxState.size.x), round(ndc.y * rtxState.size.y));
            LightSample cache = thisDirectResv[coord.y * rtxState.size.x + coord.x].lightSample;

            cache.Li *= Eval(state, -r.direction, state.ffnormal, cache.wi, dummyPdf);
            if (dummyPdf > 0)
                radiance += cache.Li * throughput;
        }
    }

    // add direct light
    float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
    if (rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
        radiance += DirectLight(state, -r.direction) * throughput / lightsourceProb;
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
    float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
    if (rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
        radiance += DirectLight(state, -r.direction) * throughput / lightsourceProb;
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
    vec3 radiance = /*state.mat.emission*/ vec3(0.0);
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
        state = GetState(r);

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
            state = GetState(r);

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

uvec4 encodeGeometryInfo(State state, float depth) {
    uvec4 gInfo;
    gInfo.x = floatBitsToUint(depth);
    gInfo.y = compress_unit_vec(state.normal);
    gInfo.z = packUnorm4x8(vec4(state.mat.metallic, state.mat.roughness, (state.mat.ior-1.0) / MAX_IOR_MINUS_ONE, state.mat.transmission));
    gInfo.w = packUnorm4x8(vec4(state.mat.albedo, 1.0)) & 0xFFFFFF; //agbr
    gInfo.w += hash8bit(state.matID);
    return gInfo;
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

vec3 PHat(Reservoir resv, State state, vec3 wo) {
    return resv.lightSample.Li * Eval(state, wo, state.ffnormal, resv.lightSample.wi, dummyPdf) *
        abs(dot(state.ffnormal, resv.lightSample.wi));
}

float BigW(Reservoir resv, State state, vec3 wo) {
    return resv.weight / (resvToScalar(PHat(resv, state, wo)) * float(resv.num));
}

bool findTemporalNeighbor(vec3 norm, float depth, float reprojDepth, uint matId, ivec2 lastCoord, out Reservoir resv) {
    int pidx = lastCoord.y * rtxState.size.x + lastCoord.x;

    vec3 pnorm; float pdepth; uint matHash;
    loadLastGeometryInfo(lastCoord, pnorm, pdepth, matHash);

    bool diff = false;
    if (lastCoord.x < 0 || lastCoord.x >= rtxState.size.x || lastCoord.x < 0 || lastCoord.y >= rtxState.size.y) {
        return false;
    }
    else if (hash8bit(matId) != matHash)
        return false;
    else if (dot(norm, pnorm) < 0.9 || reprojDepth  > pdepth * 1.05) {
        return false;
    }
    // else if (dot(norm, pnorm) < 0.9 || reprojDepth  > pdepth) {
    //     return false;
    // }
    resv = lastDirectResv[pidx];
    return true;
}

/*
* Assume temporally reused result is temporarily stored in tempDirectResv
*/
bool findSpatialNeighbor(vec3 norm, float depth, uint matId, out Reservoir resv) {
    const float Radius = 30.0;

    vec2 p = toConcentricDisk(vec2(rand(prd.seed), rand(prd.seed)));
    int px = int(float(imageCoords.x + p.x) + 0.5);
    int py = int(float(imageCoords.y + p.y) + 0.5);
    int pidx = py * rtxState.size.x + px;

    vec3 pnorm; float pdepth;
    loadLastGeometryInfo(imageCoords, pnorm, pdepth);

    bool diff = false;
    if (px < 0 || px >= rtxState.size.x || py < 0 || py >= rtxState.size.y) {
        return false;
    }
    else if (dot(norm, pnorm) < 0.1 || abs(depth - pdepth) > depth * 0.1) {
        return false;
    }
    resv = tempDirectResv[pidx];
    return true;
}

bool mergeSpatialNeighbors(vec3 norm, float depth, uint matId, out Reservoir resv) {
    bool valid = false;
    resvReset(resv);
    for (int i = 0; i < 5; i++) {
        Reservoir spatial;
        if (findSpatialNeighbor(norm, depth, matId, spatial)) {
            if (!resvInvalid(spatial)) {
                resvMerge(resv, spatial, rand(prd.seed));
                valid = true;
            }
        }
    }
    return valid;
}

void saveNewReservoir(Reservoir resv) {
    thisDirectResv[imageCoords.y * rtxState.size.x + imageCoords.x] = resv;
}

void cacheTempReservoir(Reservoir resv) {
    tempDirectResv[imageCoords.y * rtxState.size.x + imageCoords.x] = resv;
}

vec2 createMotionVector(vec3 pos) {
    vec4 proj = sceneCamera.lastProjView * vec4(pos, 1.0);
    vec3 ndc = proj.xyz / proj.w;
    return ndc.xy * 0.5 + 0.5;
}

ivec2 createMotionIndex(vec3 pos) {
    //return min(ivec2(createMotionVector(pos) * vec2(rtxState.size - 1)), rtxState.size - 1);
    return ivec2(createMotionVector(pos) * vec2(rtxState.size - 1)) + 1;
}

vec3 DirectSample(Ray r) {
    int idx = imageCoords.y * rtxState.size.x + imageCoords.x;
    uvec4 gInfo;
    ClosestHit(r);

    if (prd.hitT >= INFINITY) {

        vec3 env;
        if (_sunAndSky.in_use == 1) {
            env = sun_and_sky(_sunAndSky, r.direction);
        }
        else {
            vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
            env = texture(environmentTexture, uv).rgb;
        }
        // Done sampling return
        imageStore(thisGbuffer, imageCoords, uvec4(floatBitsToUint(INFINITY), 0, 0, 0));
        imageStore(motionVector, imageCoords, ivec4(0, 0, 0, 0));
        return (env * rtxState.hdrMultiplier);
    }
    State state = GetState(r);
    // Filling material structures
    GetMaterials(state, r);

    ivec2 motionIdx = createMotionIndex(state.position);
    imageStore(motionVector, imageCoords, ivec4(motionIdx, 0, 0));

    gInfo = encodeGeometryInfo(state, prd.hitT);
    imageStore(thisGbuffer, imageCoords, gInfo);
    barrier();

    if (rtxState.debugging_mode > eIndirectStage) {
        return DebugInfo(state);
    }

    if (state.isEmitter) {
        return state.mat.emission;
    }

    vec3 wo = -r.direction;
    vec3 direct = vec3(0.0);
    vec3 albedo = state.mat.albedo;
    state.mat.albedo = vec3(1.0);

#ifndef DIRECT_ONLY
    float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
    if (rand(prd.seed) < lightsourceProb) {
#endif
        if (rtxState.ReSTIRState == eNone) {
            direct = DirectLight(state, wo);
        }
        else {
            Reservoir resv;
            resvReset(resv);

            for (int i = 0; i < rtxState.RISRepeat; i++) {
                LightSample lsample;
                float p = SampleDirectLightNoVisibility(state.position, lsample);

                vec3 g = lsample.Li * Eval(state, wo, state.ffnormal, lsample.wi, dummyPdf) * abs(dot(state.ffnormal, lsample.wi));
                float weight = resvToScalar(g / p);

                if (IsPdfInvalid(p) || isnan(weight)) {
                    weight = 0.0;
                }
                resvUpdate(resv, lsample, weight, rand(prd.seed));
            }
            LightSample lsample = resv.lightSample;
            Ray shadowRay;
            shadowRay.origin = OffsetRay(state.position, state.ffnormal);
            shadowRay.direction = lsample.wi;

            if (Occlusion(shadowRay, state, lsample.dist)) {
                resv.weight = 0.0;
            }

            if (rtxState.ReSTIRState == eTemporal || rtxState.ReSTIRState == eSpatiotemporal) {
                //float reprojDepth = length(vec3(sceneCamera.lastView * vec4(state.position, 1.0)));
                float reprojDepth = length(sceneCamera.lastPosition - state.position);
                Reservoir temporal;
                if (findTemporalNeighbor(state.normal, prd.hitT, reprojDepth, state.matID, motionIdx, temporal)) {
                    if (!resvInvalid(temporal)) {
                        //resvPreClampedMerge20(resv, temporal, rand(prd.seed));
                        resvMerge(resv, temporal, rand(prd.seed));
                    }
                }
            }

            Reservoir tempResv = resv;

            if (rtxState.ReSTIRState == eSpatial || rtxState.ReSTIRState == eSpatiotemporal) {
                Reservoir spatial;
                resvReset(spatial);

                barrier();
                resvCheckValidity(resv);
                cacheTempReservoir(resv);
                barrier();

                Reservoir spatialAggregate;
                if (mergeSpatialNeighbors(state.normal, prd.hitT, state.matID, spatialAggregate)) {
                    if (!resvInvalid(spatialAggregate)) {
                        resvMerge(spatial, spatialAggregate, rand(prd.seed));
                    }
                }
                barrier();
                resvCheckValidity(resv);
                cacheTempReservoir(resv);
                barrier();

                if (mergeSpatialNeighbors(state.normal, prd.hitT, state.matID, spatialAggregate)) {
                    if (!resvInvalid(spatialAggregate)) {
                        resvMerge(spatial, spatialAggregate, rand(prd.seed));
                    }
                }

                if (!resvInvalid(spatial)) {
                    //resvClamp(spatial, 128);
                    resvMerge(resv, spatial, rand(prd.seed));
                }
            }
            resvCheckValidity(tempResv);
            resvClamp(tempResv, 1280);
            saveNewReservoir(tempResv);
            lsample = resv.lightSample;

            if (!resvInvalid(resv)) {
                vec3 LiBsdf = lsample.Li * Eval(state, wo, state.ffnormal, lsample.wi, dummyPdf);
                direct = LiBsdf / resvToScalar(LiBsdf) * resv.weight / float(resv.num);
            }
        }
#ifndef DIRECT_ONLY
        direct /= lightsourceProb;
    }
    else {
        Ray shadowRay;
        BsdfSampleRec bsdfSampleRec;
        bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);

        if (bsdfSampleRec.pdf > 0.0) {
            vec3 throughput = bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
            shadowRay.direction = bsdfSampleRec.L;
            shadowRay.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);
            ClosestHit(shadowRay);
            if (prd.hitT >= INFINITY) {
                vec3 env;
                if (_sunAndSky.in_use == 1)
                    env = sun_and_sky(_sunAndSky, shadowRay.direction);
                else {
                    vec2 uv = GetSphericalUv(shadowRay.direction);  // See sampling.glsl
                    env = texture(environmentTexture, uv).rgb;
                }
                // Done sampling return
                direct = (env * rtxState.hdrMultiplier) * throughput;
            }
            else {
                State state2;
                ShadeState sstate = GetShadeState(prd);
                state2.matID = sstate.matIndex;
                state2.ffnormal = dot(state2.normal, r.direction) <= 0.0 ? state2.normal : -state2.normal;
                GetMaterials(state2, shadowRay);

                direct = state2.mat.emission * throughput;
            }
        }
        direct /= 1.0 - lightsourceProb;
    }
#endif
    if (isnan(direct.x) || isnan(direct.y) || isnan(direct.z)) {
        direct = vec3(0.0);
    }
    //direct = vec3(vec2(motionIdx) / vec2(rtxState.size), 0);
    return state.mat.emission + direct * albedo;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
Ray raySpawn(ivec2 imageCoords, ivec2 sizeImage) {
    // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    // vec2 subpixel_jitter = rtxState.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(prd.seed), rand(prd.seed));

    // Compute sampling position between [-1 .. 1]
    const vec2 pixelCenter = vec2(imageCoords)/* + subpixel_jitter*/;
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
