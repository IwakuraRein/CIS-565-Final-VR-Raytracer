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
// This file has the Ray Query functions for Closest-Hit and Any-Hit shader.
// The RTX pipeline implementation of thoses functions are in traceray_rtx.
// This is used in pathtrace.glsl (Ray-Generation shader)

#include "shade_state.glsl"

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true is opaque
//----------------------------------------------------------
bool HitTest(in rayQueryEXT rayQuery, in Ray r) {
  int InstanceCustomIndexEXT = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
  int PrimitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);

  // Retrieve the Primitive mesh buffer information
  InstanceData pinfo = geoInfo[InstanceCustomIndexEXT];
  const uint matIndex = max(0, pinfo.materialIndex);  // material of primitive mesh
  GltfShadeMaterial mat = materials[matIndex];

  //// Back face culling defined by material
  //bool front_face = rayQueryGetIntersectionFrontFaceEXT(rayQuery, false);
  //if(mat.doubleSided == 0 && front_face == false)
  //{
  //  return false;
  //}

  //// Early out if there is no opacity function
  //if(mat.alphaMode == ALPHA_OPAQUE)
  //{
  //  return true;
  //}

  float baseColorAlpha = mat.pbrBaseColorFactor.a;
  if(mat.pbrBaseColorTexture > -1) {
    const uint idGeo = InstanceCustomIndexEXT;  // Geometry of this instance
    const uint idPrim = PrimitiveID;             // Triangle ID

    // Primitive buffer addresses
    Indices indices = Indices(geoInfo[idGeo].indexAddress);
    Vertices vertices = Vertices(geoInfo[idGeo].vertexAddress);

    // Indices of this triangle primitive.
    uvec3 tri = indices.i[idPrim];

    // All vertex attributes of the triangle.
    VertexAttributes attr0 = vertices.v[tri.x];
    VertexAttributes attr1 = vertices.v[tri.y];
    VertexAttributes attr2 = vertices.v[tri.z];

    // Get the texture coordinate
    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    const vec2 uv0 = attr0.texcoord;
    const vec2 uv1 = attr1.texcoord;
    const vec2 uv2 = attr2.texcoord;
    vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    // Uv Transform
    texcoord0 = (vec4(texcoord0.xy, 1, 1) * mat.uvTransform).xy;

    baseColorAlpha *= texture(texturesMap[nonuniformEXT(mat.pbrBaseColorTexture)], texcoord0).a;
  }

  float opacity;
  if(mat.alphaMode == ALPHA_MASK) {
    opacity = baseColorAlpha > mat.alphaCutoff ? 1.0 : 0.0;
  } else {
    opacity = baseColorAlpha;
  }

  // do alpha blending the stochastically way
  if(rand(prd.seed) > opacity)
    return false;

  return true;
}

int mipLevels(ivec2 extent) {
  return int(floor(log2(max(extent.x, extent.y))));
}
ivec2 getUnormCoord(vec2 uv, int mipLevel) {
  float texelSize = pow(0.5, mipLevel);
  return ivec2(int(floor(uv.x / texelSize)), int(floor(uv.y / texelSize)));
}
vec2 getUV(ivec2 unormCoord, int mipLevel) {
  float texelSize = pow(0.5, mipLevel);
  return texelSize * unormCoord + texelSize * 0.5;
}
bool pointInTrig(vec2 uv, vec2 uv0, vec2 uv1, vec2 uv2) {
  vec2 v0 = uv2 - uv0;
  vec2 v1 = uv1 - uv0;
  vec2 v2 = uv - uv0;

  float dot00 = dot(v0, v0);
  float dot01 = dot(v0, v1);
  float dot02 = dot(v0, v2);
  float dot11 = dot(v1, v1);
  float dot12 = dot(v1, v2);

  float inverDeno = 1.0 / (dot00 * dot11 - dot01 * dot01);

  float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
  if(u < 0 || u > 1) // if u out of range, return directly
  {
    return false;
  }

  float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
  if(v < 0 || v > 1) // if v out of range, return directly
  {
    return false;
  }

  return u + v <= 1;
}
bool pointInRect(float l, float r, float t, float b, vec2 p) {
  return p.x > l && p.x < r && p.y > b && p.y < t;
}
bool lineRectangleIntersect(float l, float r, float t, float b, vec2 p0, vec2 p1) {
  float minX = p0.x;
  float maxX = p1.x;

  if(p0.x > p1.x) {
    minX = p1.x;
    maxX = p0.x;
  }

  // Find the intersection of the segment's and rectangle's x-projections
  if(maxX > r)
    maxX = r;

  if(minX < l)
    minX = l;

  // If their projections do not intersect return false
  if(minX > maxX)
    return false;

  // Find corresponding min and max Y for min and max X we found before
  float minY = p0.y;
  float maxY = p1.y;

  float dx = p1.x - p0.x;

  if(abs(dx) > 1e-5) {
    float a = (p1.y - p0.y) / dx;
    float b = p0.y - a * p0.x;
    minY = a * minX + b;
    maxY = a * maxX + b;
  }

  if(minY > maxY) {
    float tmp = maxY;
    maxY = minY;
    minY = tmp;
  }

  // Find the intersection of the segment's and rectangle's y-projections
  if(maxY > t)
    maxY = t;

  if(minY < b)
    minY = b;

  // If Y-projections do not intersect return false
  if(minY > maxY)
    return false;

  return true;
}
bool overlapWithTrig(vec2 uv, int mipLevel, vec2 uv0, vec2 uv1, vec2 uv2) {
  float halfTexel = pow(0.5, mipLevel + 1);
  float l = uv.x - halfTexel;
  float r = uv.x + halfTexel;
  float t = uv.y + halfTexel;
  float b = uv.x - halfTexel;

  if(pointInRect(l, r, t, b, uv0) ||
    pointInRect(l, r, t, b, uv1) ||
    pointInRect(l, r, t, b, uv2))
    return true;

  if(lineRectangleIntersect(l, r, t, b, uv0, uv1) ||
    lineRectangleIntersect(l, r, t, b, uv0, uv2) ||
    lineRectangleIntersect(l, r, t, b, uv1, uv2))
    return true;
  return false;
}
vec3 getT(vec3 v0, vec3 v1, vec3 v2, vec2 uv0, vec2 uv1, vec2 uv2) {
  vec3 E1 = v1 - v0;
  vec3 E2 = v2 - v0;
  vec2 dUV1 = uv1 - uv0;
  vec2 dUV2 = uv2 - uv0;
  float f = 1.0 / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
  return normalize(vec3(f * (dUV2.y * E1.x - dUV1.y * E2.x), f * (dUV2.y * E1.y - dUV1.y * E2.y), f * (dUV2.y * E1.z - dUV1.y * E2.z)));
}
bool rayAABBIntersect(vec3 Min, vec3 Max, Ray r, out float dist) {
  vec3 invDir = 1.0 / r.direction;
  float tx1 = (Min.x - r.origin.x) * invDir.x;
  float tx2 = (Max.x - r.origin.x) * invDir.x;

  float tmin = min(tx1, tx2);
  float tmax = max(tx1, tx2);

  float ty1 = (Min.y - r.origin.y) * invDir.y;
  float ty2 = (Max.y - r.origin.y) * invDir.y;

  tmin = max(tmin, min(ty1, ty2));
  tmax = min(tmax, max(ty1, ty2));

  float tz1 = (Min.z - r.origin.z) * invDir.z;
  float tz2 = (Max.z - r.origin.z) * invDir.z;

  tmin = max(tmin, min(tz1, tz2));
  tmax = min(tmax, max(tz1, tz2));
  dist = tmin;
  return tmax > 0 && tmax > tmin;
}
vec3 projectVec(vec3 T, vec3 v0, vec3 v1) {
  vec3 n = normalize(cross(v0, v1));
  return T - n * dot(n, T);
}
vec3 getBarycentricCoord3D(vec3 p, vec3 v0, vec3 v1, vec3 v2) {
  vec3 vec_0 = projectVec(p - v0, v1 - v0, v2 - v0);
  vec3 projection = v0 + vec_0;
  vec3 vec_1 = v1 - projection;
  vec3 vec_2 = v2 - projection;
  float a0 = length(cross(vec_1, vec_2));
  float a1 = length(cross(vec_0, vec_2));
  float a2 = length(cross(vec_0, vec_1));
  float denom = 1.0 / (a0 + a1 + a2);
  // distToBarycenter = length(projection - (v0+v1+v2) * 0.333333333333333333333333);
  return vec3(a0 * denom, a1 * denom, a2 * denom);
}
vec3 getBarycentricCoord2D(vec2 p, vec2 v0, vec2 v1, vec2 v2) {
  vec3 vec_0 = vec3(v0 - p, 0);
  vec3 vec_1 = vec3(v1 - p, 0);
  vec3 vec_2 = vec3(v2 - p, 0);
  float a0 = length(cross(vec_1, vec_2));
  float a1 = length(cross(vec_0, vec_2));
  float a2 = length(cross(vec_0, vec_1));
  float denom = 1.0 / (a0 + a1 + a2);
  return vec3(a0 * denom, a1 * denom, a2 * denom);
}
bool displacementIntersectionTest(Ray r, rayQueryEXT rayQuery, out float hitT, out vec2 outBaryCoord) {
  const uint idGeo = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false); // Geometry of this instance
  const uint idPrim = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false); // Triangle ID
  // mat4 w2o = mat4(rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false));
  mat4 o2w = mat4(rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false));
  // mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false);
  Displacement displacement = materials[max(0, geoInfo[idGeo].materialIndex)].displacement;
  mat4 uvTransform = materials[max(0, geoInfo[idGeo].materialIndex)].uvTransform;

  // Primitive buffer addresses
  Indices indices = Indices(geoInfo[idGeo].indexAddress);
  Vertices vertices = Vertices(geoInfo[idGeo].vertexAddress);
  uvec3 tri = indices.i[idPrim];

  // All in object space
  vec3 v0 = vertices.v[tri.x].position.xyz;
  vec2 uv0 = (vec4(vertices.v[tri.x].texcoord.xy, 1, 1) * uvTransform).xy;
  vec3 v1 = vertices.v[tri.y].position.xyz;
  vec2 uv1 = (vec4(vertices.v[tri.y].texcoord.xy, 1, 1) * uvTransform).xy;
  vec3 v2 = vertices.v[tri.z].position.xyz;
  vec2 uv2 = (vec4(vertices.v[tri.z].texcoord.xy, 1, 1) * uvTransform).xy;
  vec3 N = normalize(cross(v1 - v0, v2 - v0));

  // get world space T,B,N
  vec3 T = getT(v0, v1, v2, uv0, uv1, uv2);
  T = normalize(vec3(o2w * vec4(T, 1.0)));
  N = normalize(vec3(o2w * vec4(N, 1.0)));
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);

  vec3 rayBaryCoord, intersectBaryCoord;
  { // world to uvn space

    // barycentric coordinate of ray origin's projection
    rayBaryCoord = getBarycentricCoord3D(r.origin, v0, v1, v2);

    // rotate into tangent space
    mat3 TBN_inv = transpose(mat3(T, B, N));
    r.direction += r.origin;
    r.origin = TBN_inv * r.origin;
    r.direction = normalize(TBN_inv * r.direction - r.origin);

    // barycentric coordinate of ray and UV plane's intersection
    vec2 intersection = (r.origin + r.direction * (-r.origin.z / r.direction.z)).xy;
    vec2 v0_t = (TBN_inv * v0).xy;
    vec2 v1_t = (TBN_inv * v1).xy;
    vec2 v2_t = (TBN_inv * v2).xy;
    intersectBaryCoord = getBarycentricCoord2D(intersection, v0_t, v1_t, v2_t);

    float tanTheta = r.origin.z / length(intersection - r.origin.xy);
    r.origin.x = dot(rayBaryCoord, vec3(uv0.x, uv1.x, uv2.x));
    r.origin.y = dot(rayBaryCoord, vec3(uv0.y, uv1.y, uv2.y));
    intersection.x = dot(intersectBaryCoord, vec3(uv0.x, uv1.x, uv2.x));
    intersection.y = dot(intersectBaryCoord, vec3(uv0.y, uv1.y, uv2.y));
    r.origin.z = length(intersection - r.origin.xy) * tanTheta;
  }

  //traversal
  int totalLevels = mipLevels(textureSize(minMaxTextures[displacement.texture], 0));
  int currentLevel = 0;
  ivec2 currentUnormCoord;
  bool hit;

  // first, find the smallest texel that wraps the triangle. this will be the root node
  for(; currentLevel < totalLevels; currentLevel++) {
    ivec2 c0 = getUnormCoord(uv0, currentLevel);
    ivec2 c1 = getUnormCoord(uv1, currentLevel);
    ivec2 c2 = getUnormCoord(uv2, currentLevel);
    if(c0 != c1 || c1 != c2) {
      break;
    }
    currentUnormCoord = c0;
  }
  currentUnormCoord *= 2;
  ivec2 currentPart = ivec2(0, 0);

  vec2 currentUV = getUV(currentUnormCoord + currentPart, currentLevel);
  if(overlapWithTrig(currentUV, currentLevel, uv0, uv1, uv2)) {
    vec2 minmax = texelFetch(minMaxTextures[displacement.texture], ivec2(textureSize(minMaxTextures[displacement.texture], totalLevels - currentLevel) * currentUV), totalLevels - currentLevel).xy;
    minmax = minmax * displacement.factor + displacement.offset;

    // rayAABBIntersect(currentUV, minmax, currentLevel, r, hitT);
    float halfTexel = pow(0.5, currentLevel-1);
    vec3 Min = vec3(currentUV-halfTexel, minmax.x);
    vec3 Max = vec3(currentUV+halfTexel, minmax.y);
    rayAABBIntersect(Min, Max, r, hitT);
  }
  hitT = 0.1; 
  outBaryCoord = vec2(0.5, 0.5);
  return hit;
}

//-----------------------------------------------------------------------
// Shoot a ray and return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void ClosestHit(Ray r) {
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;  // gl_RayFlagsNoneEXT
  prd.hitT = INFINITY;

  // Initializes a ray query object but does not start traversal
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery,     //
  topLevelAS,   // acceleration structure
  rayFlags,     // rayFlags
  0xFF,         // cullMask
  r.origin,     // ray origin
  0.0,          // ray min range
  r.direction,  // ray direction
  INFINITY);    // ray max range

  float displaceHitT = INFINITY + 1;
  vec2 displaceBaryCoord = vec2(0, 0);

  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery)) {
    uint primType = rayQueryGetIntersectionTypeEXT(rayQuery, false);
    if(primType == gl_RayQueryCandidateIntersectionTriangleEXT) {
      if(HitTest(rayQuery, r)) {
        rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
      }
    } else if(primType == gl_RayQueryCandidateIntersectionAABBEXT) {
      if(HitTest(rayQuery, r)) { // The hit was opaque
        float T;
        vec2 baryCoord;
        if(displacementIntersectionTest(r, rayQuery, T, baryCoord)) {
          rayQueryGenerateIntersectionEXT(rayQuery, T);
          if(T < displaceHitT) {
            displaceHitT = T;
            displaceBaryCoord = baryCoord;
          }
        }
      }
    }
  }

  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
  if(hit) {
    prd.hitT = rayQueryGetIntersectionTEXT(rayQuery, true);
    prd.primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    prd.instanceID = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    prd.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    // prd.baryCoord = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    prd.baryCoord = (prd.hitT == displaceHitT) ? displaceBaryCoord : rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    prd.objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    prd.worldToObject = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool AnyHit(Ray r, float maxDist) {
  shadow_payload.isHit = true;      // Asume hit, will be set to false if hit nothing (miss shader)
  shadow_payload.seed = prd.seed;  // don't care for the update - but won't affect the rahit shader
  uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

  // Initializes a ray query object but does not start traversal
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery,     //
  topLevelAS,   // acceleration structure
  rayFlags,     // rayFlags
  0xFF,         // cullMask
  r.origin,     // ray origin
  0.0,          // ray min range
  r.direction,  // ray direction
  maxDist);     // ray max range

  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery)) {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
      if(HitTest(rayQuery, r)) {
        rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
      }
    }
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
      // if(HitTest(rayQuery, r)) {
      //   float T;
      //   vec2 baryCoord;
      //   if (displacementIntersectionTest(r, 
      //     rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false),
      //     rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false),
      //     T, baryCoord)) {
      //     rayQueryGenerateIntersectionEXT(rayQuery, T);
      //   }
      // }
    }
  }

  // add to ray contribution from next event estimation
  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
}
