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

#include "pbr_disney.glsl"
#include "pbr_gltf.glsl"
#include "gltf_material.glsl"
#include "punctual.glsl"
#include "env_sampling.glsl"
#include "shade_state.glsl"
#include "restir.glsl"

const float InvalidPdf = -1.0;
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
    state.isEmitter = false;
    state.specularBounce = false;
    state.isSubsurface = false;
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
  if(rtxState.pbrMode == 0)
    return DisneyEval(state, V, N, L, pdf);
  else
    return PbrEval(state, V, N, L, pdf);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 Sample(in State state, in vec3 V, in vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed) {
  if(rtxState.pbrMode == 0)
    return DisneySample(state, V, N, L, pdf, seed);
  else
    return PbrSample(state, V, N, L, pdf, seed);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 DebugInfo(in State state) {
  switch(rtxState.debugging_mode) {
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
    case eAlpha:
      return vec3(state.mat.alpha);
    case eRoughness:
      return vec3(state.mat.roughness);
    case eTexcoord:
      return vec3(state.texCoord, 0);
    case eTangent:
      return vec3(state.tangent.xyz + vec3(1)) * .5;
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
    vec4 dirAndPdf;

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
    float dist = length(y - x);
    lightSample.Li = emission / area;
    lightSample.wi = normalize(y - x);
    lightSample.dist = dist;
    return light.impSamp.pdf * (dist * dist) / (area * abs(dot(dirAndPdf.xyz, normal)));
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
    float dist = length(light.position - x);
    lightSample.Li = light.color * light.intensity / (dist * dist);
    lightSample.wi = normalize(light.position - x);
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
        lightSample.dist = 1e24;
        return dirAndPdf.w * rtxState.environmentProb;
    }
    else {
        float pdf;
        if (rnd < rtxState.environmentProb + (1.0 - rtxState.environmentProb) * lightBufInfo.trigSampProb) {
            // Sample triangle mesh light
            pdf = SampleTriangleLight(pos, lightSample) * lightBufInfo.trigSampProb;
        }
        else {
            // Sample point light
            pdf = SamplePuncLight(pos, lightSample) * (1.0 - lightBufInfo.trigSampProb);
        }
        return pdf * (1.0 - rtxState.environmentProb);
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

vec3 DirectLuminance(in Ray r, in State state, out vec3 luminance, out vec3 dir) { // importance sample on light sources
    return DirectLight(state, -r.direction);
}


bool UpdateSample(inout Ray r, in State state, in float screenDepth, inout vec3 radiance, inout vec3 throughput, inout vec3 absorption) {

    // Reset absorption when ray is going out of surface
  if(dot(state.normal, state.ffnormal) > 0.0) {
    absorption = vec3(0.0);
  }

  // Emissive material
  radiance += state.mat.emission * throughput;
  // add screenspace indirect
  vec3 ndc = vec3(sceneCamera.projView * vec4(state.position, 1.0));
  ndc += vec3(1, 1, 0);
  ndc *= 0.5;
  //check if this sample appears in direct stage's result
  if(ndc.x >= 0.0 && ndc.x <= 1.0 && ndc.y >= 0.0 && ndc.y <= 1.0) {
    // TODO: adjust offset according to normal
    if(getDepth(ndc.z) <= (screenDepth + 1e-3)) {
      ivec2 coord = ivec2(round(ndc.x * rtxState.size.x), round(ndc.y * rtxState.size.y));
      uvec2 cache = imageLoad(thisDirectCache, coord).xy;
      vec3 Li = unpackUnormYCbCr(cache.x);
      vec3 dir = decompress_unit_vec(cache.y);

      Li *= Eval(state, -r.direction, state.ffnormal, dir, dummyPdf);
      radiance += Li * throughput;
    }
  }

  // KHR_materials_unlit
  if(state.mat.unlit) {
    radiance += state.mat.albedo * throughput;
    return false;
  }

  // Add absoption (transmission / volume)
  throughput *= exp(-absorption * prd.hitT);

  // add direct light
  float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
  if(rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
    radiance += DirectLight(state, -r.direction) * throughput / lightsourceProb;
  }

  BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
  bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
  bsdfSampleRec.L = normalize(bsdfSampleRec.L);

    // Set absorption only if the ray is currently inside the object.
  if(dot(state.ffnormal, bsdfSampleRec.L) < 0.0) {
    absorption = -log(state.mat.attenuationColor) / vec3(state.mat.attenuationDistance);
  }

  if(bsdfSampleRec.pdf > 0.0) {
    throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
  } else {
    return false;
  }

  // Next ray
  r.direction = bsdfSampleRec.L;
  r.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

  return true;
}
bool UpdateSampleWithoutEmission(inout Ray r, in State state, inout vec3 radiance, inout vec3 throughput, inout vec3 absorption) {

  // Reset absorption when ray is going out of surface
  if(dot(state.normal, state.ffnormal) > 0.0) {
    absorption = vec3(0.0);
  }

  // KHR_materials_unlit
  if(state.mat.unlit) {
    radiance += state.mat.albedo * throughput;
    return false;
  }

  // Add absoption (transmission / volume)
  throughput *= exp(-absorption * prd.hitT);

  // add direct light
  float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
  if(rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
    radiance += DirectLight(state, -r.direction) * throughput / lightsourceProb;
  }

  BsdfSampleRec bsdfSampleRec;
  // Sampling for the next ray
  bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
  bsdfSampleRec.L = normalize(bsdfSampleRec.L);
    // Set absorption only if the ray is currently inside the object.
  if(dot(state.ffnormal, bsdfSampleRec.L) < 0.0) {
    absorption = -log(state.mat.attenuationColor) / vec3(state.mat.attenuationDistance);
  }

  if(bsdfSampleRec.pdf > 0.0) {
    throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
  } else {
    return false;
  }

  // Next ray
  r.direction = bsdfSampleRec.L;
  r.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

  return true;
}
bool UpdateSampleWithoutEmissionDirectLight(inout Ray r, in State state, inout vec3 radiance, inout vec3 throughput, inout vec3 absorption) {

    // Reset absorption when ray is going out of surface
  if(dot(state.normal, state.ffnormal) > 0.0) {
    absorption = vec3(0.0);
  }

    // KHR_materials_unlit
  if(state.mat.unlit) {
    radiance += state.mat.albedo * throughput;
    return false;
  }

    // Add absoption (transmission / volume)
  throughput *= exp(-absorption * prd.hitT);

  BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
  bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);
  bsdfSampleRec.L = normalize(bsdfSampleRec.L);

    // Set absorption only if the ray is currently inside the object.
  if(dot(state.ffnormal, bsdfSampleRec.L) < 0.0) {
    absorption = -log(state.mat.attenuationColor) / vec3(state.mat.attenuationDistance);
  }

  if(bsdfSampleRec.pdf > 0.0) {
    throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
  } else {
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
  vec3 absorption = vec3(0.0);

  { // first intersection
    // we don't want to reintroduce the luminance that direct stage has already done
    if(!UpdateSampleWithoutEmissionDirectLight(r, state, radiance, throughput, absorption))
      return radiance;
#ifdef RR
    // For Russian-Roulette (minimizing live state)
    float rrPcont = (0 >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
    if(rand(prd.seed) >= rrPcont)
      return radiance;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
  }

  { // second intersection
    ClosestHit(r);

    // Hitting the environment
    if(prd.hitT >= INFINITY) {
      return radiance;
    }

    // Get Position, Normal, Tangents, Texture Coordinates, Color
    ShadeState sstate = GetShadeState(prd);
    state.position = sstate.position;
    state.normal = sstate.normal;
    state.tangent = sstate.tangent_u[0];
    state.bitangent = sstate.tangent_v[0];
    state.texCoord = sstate.text_coords[0];
    state.matID = sstate.matIndex;
    state.position = sstate.position;
    state.isEmitter = false;
    state.specularBounce = false;
    state.isSubsurface = false;
    state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;

    // Filling material structures
    GetMaterialsAndTextures(state, r);
    // we don't use vertex color cause there isn't enough room for it in the Gbuffer
    // state.mat.albedo *= sstate.color;

    // Save as above. we don't want to reintroduce the first direct lighting
    // but second direct lighting is allowed
    if(!UpdateSampleWithoutEmission(r, state, radiance, throughput, absorption))
      return radiance;

#ifdef RR
    // For Russian-Roulette (minimizing live state)
    float rrPcont = (1 >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
    if(rand(prd.seed) >= rrPcont)
      return radiance;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
  }

  { // second bounce and more
    for(int depth = 2; depth < rtxState.maxDepth; depth++) {
      ClosestHit(r);

      // Hitting the environment
      if(prd.hitT >= INFINITY) {
        vec3 env;
        if(_sunAndSky.in_use == 1)
          env = sun_and_sky(_sunAndSky, r.direction);
        else {
          vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
          env = texture(environmentTexture, uv).rgb;
        }
        // Done sampling return
        return radiance + (env * rtxState.hdrMultiplier * throughput);
      }

      // Get Position, Normal, Tangents, Texture Coordinates, Color
      ShadeState sstate = GetShadeState(prd);
      state.position = sstate.position;
      state.normal = sstate.normal;
      state.tangent = sstate.tangent_u[0];
      state.bitangent = sstate.tangent_v[0];
      state.texCoord = sstate.text_coords[0];
      state.matID = sstate.matIndex;
      state.position = sstate.position;
      state.isEmitter = false;
      state.specularBounce = false;
      state.isSubsurface = false;
      state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;

       // Filling material structures
      GetMaterialsAndTextures(state, r);
      // Color at vertices
      // state.mat.albedo *= sstate.color;

      if(!UpdateSample(r, state, hitT, radiance, throughput, absorption))
        return radiance;

  #ifdef RR
      // For Russian-Roulette (minimizing live state)
      float rrPcont = (depth >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
      if(rand(prd.seed) >= rrPcont)
        return radiance;                // paths with low throughput that won't contribute
      throughput /= rrPcont;  // boost the energy of the non-terminated paths
  #endif
    }
  }

  return radiance;
}

/*
vec3 DirectSample(Ray r, out uvec4 gInfo, inout Reservoir resv) {
  // for (int id = 0; id < lightBufInfo.trigLightSize; id++){
  //   TrigLight light = trigLights[id];
  //   vec3 v0 = light.v0;
  //   vec3 v1 = light.v1;
  //   vec3 v2 = light.v2;

  //   vec3 r0 = v0-r.origin;
  //   vec3 r1 = v1-r.origin;
  //   vec3 r2 = v2-r.origin;

  //   float cos0 = dot(normalize(r0), r.direction);
  //   float cos1 = dot(normalize(r1), r.direction);
  //   float cos2 = dot(normalize(r2), r.direction);

  //   float sin0 = sqrt(1-cos0*cos0);
  //   float sin1 = sqrt(1-cos1*cos1);
  //   float sin2 = sqrt(1-cos2*cos2);

  //   if (length(r0)*sin0 <= 0.1) return vec3(0, 1, 0);
  //   if (length(r1)*sin1 <= 0.1) return vec3(0, 1, 0);
  //   if (length(r2)*sin2 <= 0.1) return vec3(0, 1, 0);
  // }
  
  resv.lightSample.Li = vec3(0);
  ClosestHit(r);
  gInfo.w = floatBitsToUint(prd.hitT);
  if(prd.hitT >= INFINITY) {
    // state.position = vec3(INFINITY) + abs(r.origin);

    vec3 env;
    if(_sunAndSky.in_use == 1)
      env = sun_and_sky(_sunAndSky, r.direction);
    else {
      vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
      env = texture(environmentTexture, uv).rgb;
    }
    // Done sampling return
    
    return (env * rtxState.hdrMultiplier);
  }
  State state;
  ShadeState sstate = GetShadeState(prd);
  state.position = sstate.position;
  state.normal = sstate.normal;
  state.tangent = sstate.tangent_u[0];
  state.bitangent = sstate.tangent_v[0];
  state.texCoord = sstate.text_coords[0];
  state.matID = sstate.matIndex;
  state.isEmitter = false;
  state.specularBounce = false;
  state.isSubsurface = false;
  state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;


  // Filling material structures
  GetMaterialsAndTextures(state, r);

  // Color at vertices
  // state.mat.albedo *= sstate.color;
  // Normal, Tangent, TexCoord, Material ID
  gInfo.y = compress_unit_vec(state.normal);
  gInfo.z = packUnorm2x16(state.texCoord);
  gInfo.x = packTangent(state.normal, state.tangent);
  gInfo.x = (state.matID << 16) + (gInfo.x << 16 >> 16);

  if(rtxState.debugging_mode > eIndirectStage)
    return DebugInfo(state);
    
  if(state.mat.unlit) {
    return state.mat.emission + state.mat.albedo;
  }

  // if roughness > 0.5, sample light sources only
  // else let random number decide
  float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
  if(rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
    return state.mat.emission + DirectLuminance(r, state, resv.lightSample.Li, resv.lightSample.wi) / lightsourceProb;
  } else { // importance sampling on brdf
    Ray shadowRay;
    BsdfSampleRec bsdfSampleRec;
    bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);

    if(bsdfSampleRec.pdf > 0.0) {
      vec3 throughput = bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
      shadowRay.direction = bsdfSampleRec.L;
      shadowRay.origin = OffsetRay(state.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);
      ClosestHit(shadowRay);
      if(prd.hitT >= INFINITY) {
        vec3 env;
        if(_sunAndSky.in_use == 1)
          env = sun_and_sky(_sunAndSky, shadowRay.direction);
        else {
          vec2 uv = GetSphericalUv(shadowRay.direction);  // See sampling.glsl
          env = texture(environmentTexture, uv).rgb;
        }
        // Done sampling return
        return state.mat.emission + (env * rtxState.hdrMultiplier) * throughput;
      }
      State state2;
      sstate = GetShadeState(prd);
      // state2.position = sstate.position;
      // state2.normal = sstate.normal;
      // state2.tangent = sstate.tangent_u[0];
      // state2.bitangent = sstate.tangent_v[0];
      // state2.texCoord = sstate.text_coords[0];
      state2.matID = sstate.matIndex;
      // state2.isEmitter = false;
      // state2.specularBounce = false;
      // state2.isSubsurface = false;
      state2.ffnormal = dot(state2.normal, r.direction) <= 0.0 ? state2.normal : -state2.normal;
      GetMaterialsAndTextures(state2, shadowRay);

      return state.mat.emission + state2.mat.emission * throughput / (1.0 - lightsourceProb);
    } else {
      return state.mat.emission;
    }
  }
}
*/

vec3 PHat(Reservoir resv, State state, vec3 wo) {
    return resv.lightSample.Li * Eval(state, wo, state.ffnormal, resv.lightSample.wi, dummyPdf) *
        abs(dot(state.ffnormal, resv.lightSample.wi));
}

float BigW(Reservoir resv, State state, vec3 wo) {
    return resv.weight / (resvToScalar(PHat(resv, state, wo)) * float(resv.num));
}

vec3 DirectSample(Ray r, out uvec4 gInfo, inout Reservoir resv) {
    resv.lightSample.Li = vec3(0);
    ClosestHit(r);
    gInfo.w = floatBitsToUint(prd.hitT);
    if (prd.hitT >= INFINITY) {
        // state.position = vec3(INFINITY) + abs(r.origin);

        vec3 env;
        if (_sunAndSky.in_use == 1)
            env = sun_and_sky(_sunAndSky, r.direction);
        else {
            vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
            env = texture(environmentTexture, uv).rgb;
        }
        // Done sampling return

        return (env * rtxState.hdrMultiplier);
    }
    State state = GetState(r);
    // Filling material structures
    GetMaterialsAndTextures(state, r);

    // Color at vertices
    // state.mat.albedo *= sstate.color;
    // Normal, Tangent, TexCoord, Material ID
    gInfo.y = compress_unit_vec(state.normal);
    gInfo.z = packUnorm2x16(state.texCoord);
    gInfo.x = packTangent(state.normal, state.tangent);
    gInfo.x = (state.matID << 16) + (gInfo.x << 16 >> 16);

    if (rtxState.debugging_mode > eIndirectStage)
        return DebugInfo(state);

    if (state.mat.unlit) {
        return state.mat.emission + state.mat.albedo;
    }
    vec3 wo = -r.direction;

    //vec3 direct= DirectLight(state, -r.direction);
    vec3 direct = vec3(0.0);

    if (rtxState.restirState == eNone) {
        direct = DirectLight(state, wo);
    }
    else {
        resvReset(resv);
        for (int i = 0; i < ReservoirSize; i++) {
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

        if (!resvInvalid(resv)) {
            direct = lsample.Li * Eval(state, wo, state.ffnormal, lsample.wi, dummyPdf) *
                abs(dot(state.ffnormal, lsample.wi)) * BigW(resv, state, wo);
        }
    }

    return state.mat.emission + direct;
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
  vec3 focalPoint = sceneCamera.focalDist * direction.xyz;
  float cam_r1 = rand(prd.seed) * M_TWO_PI;
  float cam_r2 = rand(prd.seed) * sceneCamera.aperture;
  vec4 cam_right = sceneCamera.viewInverse * vec4(1, 0, 0, 0);
  vec4 cam_up = sceneCamera.viewInverse * vec4(0, 1, 0, 0);
  vec3 randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
  vec3 finalRayDir = normalize(focalPoint - randomAperturePos);

  return Ray(origin.xyz + randomAperturePos, finalRayDir);
}
