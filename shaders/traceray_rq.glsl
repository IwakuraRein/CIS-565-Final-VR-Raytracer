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

bool displacementIntersectionTest(Ray r, int instanceID, int primitiveID, out float T, out vec2 baryCoord) {
  const uint idGeo = instanceID;  // Geometry of this instance
  const uint idPrim = primitiveID;             // Triangle ID

  // Primitive buffer addresses
  Indices  indices  = Indices(geoInfo[idGeo].indexAddress);
  Vertices vertices = Vertices(geoInfo[idGeo].vertexAddress);
  uvec3 tri = indices.i[idPrim];

  // All vertex attributes of the triangle.
  vec3 v0 = vertices.v[tri.x].position.xyz;
  vec3 v1 = vertices.v[tri.y].position.xyz;
  vec3 v2 = vertices.v[tri.z].position.xyz;


    // compute plane's normal
  vec3 v0v1 = v1 - v0;
  vec3 v0v2 = v2 - v0; 
    // no need to normalize
  vec3 N = cross(v0v1, v0v2);  //N 
  float area2 = length(N); 

    // Step 1: finding P

    // check if ray and plane are parallel.
  float NdotRayDirection = dot(N, r.direction);
  if(abs(NdotRayDirection) < 1e-5)  //almost 0 
    return false;  //they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
  float d = dot(-N, v0); 

    // compute t (equation 3)
  T = -(dot(N, r.origin) + d) / NdotRayDirection; 

    // check if the triangle is in behind the ray
  if(T < 0)
    return false;  //the triangle is behind 

    // compute the intersection point using equation 1
  vec3 P = r.origin + T * r.direction; 

    // Step 2: inside-outside test
  vec3 C;  //vector perpendicular to triangle's plane 

    // edge 0
  vec3 edge0 = v1 - v0;
  vec3 vp0 = P - v0;
  C = cross(edge0, vp0);
  if(dot(N, C) < 0)
    return false;  //P is on the right side 

    // edge 1
  vec3 edge1 = v2 - v1;
  vec3 vp1 = P - v1;
  C = cross(edge1, vp1);
  if(dot(N, C) < 0)
    return false;  //P is on the right side 

    // edge 2
  vec3 edge2 = v0 - v2;
  vec3 vp2 = P - v2;
  C = cross(edge2, vp2);
  if(dot(N, C) < 0)
    return false;  //P is on the right side; 

  return true;  //this ray hits the triangle 
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

  float displaceHitT = INFINITY;
  vec2 displaceBaryCoord = vec2(0, 0);

  // Start traversal: return false if traversal is complete
  while(rayQueryProceedEXT(rayQuery)) {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
      if(HitTest(rayQuery, r)) {
        rayQueryConfirmIntersectionEXT(rayQuery);  // The hit was opaque
      }
    }
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
      // if(HitTest(rayQuery, r)) { // The hit was opaque
      //   float T;
      //   float baryCoord;
      //   if (displacementIntersectionTest(r, 
      //     rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false),
      //     rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false),
      //     T, baryCoord)) {
      //     rayQueryGenerateIntersectionEXT(rayQuery, T);
      //     if (T < displaceHitT) { displaceHitT = T; displaceBaryCoord = baryCoord; }
      //   }
      // }
    }
  }

  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
  if(hit) {
    prd.hitT = rayQueryGetIntersectionTEXT(rayQuery, true);
    prd.primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    prd.instanceID = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    prd.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    // prd.baryCoord = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    prd.baryCoord = (prd.hitT == displaceHitT)? displaceBaryCoord : rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
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
