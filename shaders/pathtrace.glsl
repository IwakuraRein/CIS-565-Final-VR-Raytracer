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
// Use for light/env contribution
struct VisibilityContribution {
  vec3 radiance;   // Radiance at the point if light is visible
  vec3 lightDir;   // Direction to the light, to shoot shadow ray
  float lightDist;  // Distance to the light (1e32 for infinite or sky)
  bool visible;    // true if in front of the face and should shoot shadow ray
};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
// VisibilityContribution DirectLight(in Ray r, in State state) {
//   vec3 Li = vec3(0);
//   float lightPdf;
//   vec3 lightContrib;
//   vec3 lightDir;
//   float lightDist = 1e32;
//   bool isLight = false;

//   VisibilityContribution contrib;
//   contrib.radiance = vec3(0);
//   contrib.visible = false;

//   // keep it simple and use either point light or environment light, each with the same
//   // probability. If the environment factor is zero, we always use the point light
//   // Note: see also miss shader
//   float p_select_light = rtxState.hdrMultiplier > 0.0f ? 0.5f : 1.0f;

//   // in general, you would select the light depending on the importance of it
//   // e.g. by incorporating their luminance

//   // Point lights
//   if(sceneCamera.nbLights != 0 && rand(prd.seed) <= p_select_light) {
//     isLight = true;

//     // randomly select one of the lights
//     int light_index = int(min(rand(prd.seed) * sceneCamera.nbLights, sceneCamera.nbLights));
//     PuncLight light = puncLights[light_index];

//     vec3 pointToLight = -light.direction;
//     float rangeAttenuation = 1.0;
//     float spotAttenuation = 1.0;

//     if(light.type != LightType_Directional) {
//       pointToLight = light.position - state.position;
//     }

//     lightDist = length(pointToLight);

//     // Compute range and spot light attenuation.
//     if(light.type != LightType_Directional) {
//       rangeAttenuation = getRangeAttenuation(light.range, lightDist);
//     }
//     if(light.type == LightType_Spot) {
//       spotAttenuation = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
//     }

//     vec3 intensity = rangeAttenuation * spotAttenuation * light.intensity * light.color;

//     lightContrib = intensity;
//     lightDir = normalize(pointToLight);
//     lightPdf = 1.0;
//   }
//   // Environment Light
//   else {
//     vec4 dirPdf = EnvSample(lightContrib);
//     lightDir = dirPdf.xyz;
//     lightPdf = dirPdf.w;
//   }

//   if(state.isSubsurface || dot(lightDir, state.ffnormal) > 0.0) {
//     // We should shoot a ray toward the environment and check if it is not
//     // occluded by an object before doing the following,
//     // but the call to traceRayEXT have to store
//     // live states (ex: sstate), which is really costly. So we will do
//     // all the computation, but adding the contribution at the end of the
//     // shader.
//     // See: https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/
//     {
//       BsdfSampleRec bsdfSampleRec;

//       bsdfSampleRec.f = Eval(state, -r.direction, state.ffnormal, lightDir, bsdfSampleRec.pdf);

//       float misWeight = isLight ? 1.0 : max(0.0, powerHeuristic(lightPdf, bsdfSampleRec.pdf));

//       Li += misWeight * bsdfSampleRec.f * abs(dot(lightDir, state.ffnormal)) * lightContrib / lightPdf;
//     }

//     contrib.visible = true;
//     contrib.lightDir = lightDir;
//     contrib.lightDist = lightDist;
//     contrib.radiance = Li;
//   }

//   return contrib;
// }

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

vec4 SampleTriangleLight(vec3 x, out vec3 radiance, out float dist) {
  if(lightBufInfo.trigLightSize == 0)
    return vec4(-1.0);

  int id = min(int(float(lightBufInfo.trigLightSize) * rand(prd.seed)), int(lightBufInfo.trigLightSize) - 1);

  if(rand(prd.seed) > trigLights[id].impSamp.q)
    id = trigLights[id].impSamp.alias;

  TrigLight light = trigLights[id];
  vec4 dirAndPdf;

  // vec4 tmp0 = (trigLightTransforms[light.transformIndex] * vec4(light.v0, 1.0));
  // vec4 tmp1 = (trigLightTransforms[light.transformIndex] * vec4(light.v1, 1.0));
  // vec4 tmp2 = (trigLightTransforms[light.transformIndex] * vec4(light.v2, 1.0));
  // vec3 v0 = tmp0.xyz / tmp0.w;
  // vec3 v1 = tmp1.xyz / tmp1.w;
  // vec3 v2 = tmp2.xyz / tmp2.w;

  // vec3 v0 = vec3(trigLightTransforms[light.transformIndex] * vec4(light.v0, 1.0));
  // vec3 v1 = vec3(trigLightTransforms[light.transformIndex] * vec4(light.v1, 1.0));
  // vec3 v2 = vec3(trigLightTransforms[light.transformIndex] * vec4(light.v2, 1.0));

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
  if(mat.emissiveTexture > -1) {
    vec2 uv = baryCoord.x * light.uv0 + baryCoord.y * light.uv1 + (1 - baryCoord.x - baryCoord.y) * light.uv2;
    emission *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(mat.emissiveTexture)], uv, 0)).rgb;
  }
  dirAndPdf.xyz = normalize(y - x);
  dist = length(y - x);
  dirAndPdf.w = light.impSamp.pdf * dist * dist / (area * abs(dot(dirAndPdf.xyz, normal)));
  radiance = emission / area;
  // radiance = emission;
  return dirAndPdf;
}

vec4 SamplePuncLight(vec3 x, out vec3 radiance, out float dist) {
  if(lightBufInfo.puncLightSize == 0)
    return vec4(-1.0);

  int id = min(int(float(lightBufInfo.trigLightSize) * rand(prd.seed)), int(lightBufInfo.puncLightSize) - 1);

  if(rand(prd.seed) > puncLights[id].impSamp.q)
    id = puncLights[id].impSamp.alias;

  PuncLight light = puncLights[id];
  vec4 dirAndPdf;
  dirAndPdf.xyz = normalize(light.position - x);
  dirAndPdf.w = light.impSamp.pdf;
  dist = length(light.position - x);
  radiance = light.color * light.intensity / (dist * dist);
  return dirAndPdf;
}

vec3 DirectLuminance(in Ray r, in State state, out uint luminance, out uint dir) { // importance sample on light sources
  vec4 dirAndPdf;
  luminance = 0;
  vec3 Li = vec3(0.0);
  float dist = INFINITY;
  float rnd = rand(prd.seed);
  if(rnd < rtxState.environmentProb) {
        // Sample environment
    dirAndPdf = EnvSample(Li);
    if(dirAndPdf.w <= 0.0)
      return vec3(0.0);
    dirAndPdf.w *= rtxState.environmentProb;
  } else {
    if(rnd < rtxState.environmentProb + (1.0 - rtxState.environmentProb) * lightBufInfo.trigSampProb) {
          // Sample triangle mesh light
      dirAndPdf = SampleTriangleLight(state.position, Li, dist);
      dirAndPdf.w *= lightBufInfo.trigSampProb;
    } else {
          // Sample point light
      dirAndPdf = SamplePuncLight(state.position, Li, dist);
      dirAndPdf.w *= 1.0 - lightBufInfo.trigSampProb;
    }
    if(dirAndPdf.w <= 0.0)
      return vec3(0.0);

    dirAndPdf.w *= (1.0 - rtxState.environmentProb);
  }
  Ray shadowRay;
  BsdfSampleRec bsdfSampleRec;
  shadowRay.origin = OffsetRay(state.position, state.ffnormal);
  shadowRay.direction = dirAndPdf.xyz;

  if(AnyHit(shadowRay, dist - abs(shadowRay.origin.x - state.position.x) -
    abs(shadowRay.origin.y - state.position.y) -
    abs(shadowRay.origin.z - state.position.z)))
    return vec3(0.0);
  else {
    bsdfSampleRec.f = Eval(state, -r.direction, state.ffnormal, shadowRay.direction, bsdfSampleRec.pdf);
    dir = compress_unit_vec(dirAndPdf.xyz);
    luminance = packUnormYCbCr(Li);
    return Li * bsdfSampleRec.f *
      max(dot(state.ffnormal, dirAndPdf.xyz), 0.0) / dirAndPdf.w;
  }
}

vec3 DirectLight(in Ray r, in State state) { // importance sample on light sources
  vec4 dirAndPdf;
  vec3 Li = vec3(0.0);
  float dist = INFINITY;
  float rnd = rand(prd.seed);
  if(rnd < rtxState.environmentProb) {
        // Sample environment
    dirAndPdf = EnvSample(Li);
    if(dirAndPdf.w <= 0.0)
      return vec3(0.0);
    dirAndPdf.w *= rtxState.environmentProb;
  } else {
    if(rnd < rtxState.environmentProb + (1.0 - rtxState.environmentProb) * lightBufInfo.trigSampProb) {
      // Sample triangle mesh light
      dirAndPdf = SampleTriangleLight(state.position, Li, dist);
      dirAndPdf.w *= lightBufInfo.trigSampProb;
    } else {
      // Sample point light
      dirAndPdf = SamplePuncLight(state.position, Li, dist);
      dirAndPdf.w *= 1.0 - lightBufInfo.trigSampProb;
    }
    if(dirAndPdf.w <= 0.0)
      return vec3(0.0);

    dirAndPdf.w *= (1.0 - rtxState.environmentProb);
  }
  Ray shadowRay;
  BsdfSampleRec bsdfSampleRec;
  shadowRay.origin = OffsetRay(state.position, state.ffnormal);
  shadowRay.direction = dirAndPdf.xyz;

  if(AnyHit(shadowRay, dist - abs(shadowRay.origin.x - state.position.x) -
    abs(shadowRay.origin.y - state.position.y) -
    abs(shadowRay.origin.z - state.position.z)))
    return vec3(0.0);
  else {
    bsdfSampleRec.f = Eval(state, -r.direction, state.ffnormal, shadowRay.direction, bsdfSampleRec.pdf);
    return Li * bsdfSampleRec.f *
      max(dot(state.ffnormal, dirAndPdf.xyz), 0.0) / dirAndPdf.w;
  }
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
  //check if this sample appears in direct stage's result
  if(ndc.x >= -1.0 && ndc.x <= 1.0 && ndc.y >= -1.0 && ndc.y <= 1.0) {
    // TODO: adjust offset according to normal
    if(getDepth(ndc.z) <= (screenDepth + 1e-3)) {
      ivec2 coord = ivec2(round(ndc.x * rtxState.size.x), round(ndc.y * rtxState.size.y));
      uvec2 cache = imageLoad(thisDirectCache, coord).xy;
      vec3 Li = unpackUnormYCbCr(cache.x);
      vec3 dir = decompress_unit_vec(cache.y);
      float dummyPdf;

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
    radiance += DirectLight(r, state) * throughput / lightsourceProb;
  }

  BsdfSampleRec bsdfSampleRec;
    // Sampling for the next ray
  bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);

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
    radiance += DirectLight(r, state) * throughput / lightsourceProb;
  }

  BsdfSampleRec bsdfSampleRec;
  // Sampling for the next ray
  bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);

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

vec3 IndirectSample(Ray r, State state, float hitT, out vec3 weight, out vec3 dir) {
  if(hitT >= INFINITY)
    return vec3(0.0);
  prd.hitT = hitT;
  vec3 radiance = /*state.mat.emission*/ vec3(0.0);
  weight = vec3(1.0);
  vec3 throughput = vec3(1.0);
  vec3 absorption = vec3(0.0);

  { // first intersection
    // we don't want to reintroduce the luminance that direct stage has already done
    if(!UpdateSampleWithoutEmissionDirectLight(r, state, radiance, throughput, absorption)) {
      return radiance;
    }
    dir = r.direction;
    weight = throughput;
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

vec3 DirectSample(Ray r, out float firstHitT, out uint Li, out uint L_dir) {
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
  Li = 0;
  L_dir = 0;
  ClosestHit(r);
  firstHitT = prd.hitT;
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
  imageStore(thisGbuffer, imageCoords, uvec4(compress_unit_vec(state.normal), compress_unit_vec(state.tangent), packUnorm2x16(state.texCoord), state.matID));

  if(rtxState.debugging_mode > eIndirectStage)
    return DebugInfo(state);

  if(state.mat.unlit) {
    return state.mat.emission + state.mat.albedo;
  }

  // if roughness > 0.5, sample light sources only
  // else let random number decide
  float lightsourceProb = min(state.mat.roughness * 2.0, 1.0);
  if(rand(prd.seed) < lightsourceProb) { // importance sampling on light sources
    return state.mat.emission + DirectLuminance(r, state, Li, L_dir) / lightsourceProb;
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
