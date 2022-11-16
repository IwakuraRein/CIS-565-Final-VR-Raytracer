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


 /*
   Various structure used by CPP and GLSL
 */


#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE


#ifdef __cplusplus
#include <stdint.h>
#include "nvmath/nvmath.h"
 // GLSL Type
using ivec2 = nvmath::vec2i;
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus  // Descriptor binding helper for C++ and GLSL
#define START_ENUM(a)                                                                                               \
  enum a                                                                                                               \
  {
#define END_ENUM() }
#else
#define START_ENUM(a) const uint
#define END_ENUM()
#endif

// Sets
START_ENUM(SetBindings)
S_ACCEL = 0,  // Acceleration structure
S_OUT = 1,  // Offscreen output image
S_SCENE = 2,  // Scene data
S_ENV = 3,  // Environment / Sun & Sky
S_RAYQ = 4, // Ray query renderer
S_WF = 5   // Wavefront extra data
END_ENUM();

// Acceleration Structure - Set 0
START_ENUM(AccelBindings)
eTlas = 0
END_ENUM();

// Output image - Set 1
START_ENUM(OutputBindings)
eDirectSampler = 0,  // As sampler
eIndirectSampler = 1,  // As sampler
eLastDirectResult = 2,   // As storage
eLastIndirectResult = 3,   // As storage
eThisDirectResult = 4,   // As storage
eThisIndirectResult = 5   // As storage
END_ENUM();

// Scene Data - Set 2
START_ENUM(SceneBindings)
eCamera = 0,
eMaterials = 1,
eInstData = 2,
ePuncLights = 3,
eTrigLights = 4,
eLightBufInfo = 5,
eTextures = 6  // must be last elem            
END_ENUM();

// Environment - Set 3
START_ENUM(EnvBindings)
eSunSky = 0,
eHdr = 1,
eImpSamples = 2
END_ENUM();

// Ray Query - Set 4
START_ENUM(RayQBindings)
eLastGbuffer = 0,
eThisGbuffer = 1,
eLastDirectCache = 2,
eLastIndirectCache = 3,
eThisDirectCache = 4,
eThisIndirectCache = 5
END_ENUM();

START_ENUM(DebugMode)
eNoDebug = 0,   //
eDirectStage = 1, //
eIndirectStage = 2, //
eBaseColor = 3,   //
eNormal = 4,   //
eDepth = 5,    //
eMetallic = 6,   //
eEmissive = 7,   //
eAlpha = 8,   //
eRoughness = 9,   //
eTexcoord = 10,   //
eTangent = 11   //
// eHeatmap = 11   //
END_ENUM();
// clang-format on

// Camera of the scene
#define CAMERA_NEAR 0.001f
#define CAMERA_FAR 100000.0f
struct SceneCamera
{
	mat4  viewInverse;
	mat4  projInverse;
	mat4  projView;
	float focalDist;
	float aperture;
	// Extra
	int nbLights;
};

struct VertexAttributes
{
	vec3 position;
	uint normal;    // compressed using oct
	vec2 texcoord;  // Tangent handiness, stored in LSB of .y
	uint tangent;   // compressed using oct
	uint color;     // RGBA
};


// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2
struct GltfShadeMaterial
{
	// 0
	vec4 pbrBaseColorFactor;
	// 4
	int   pbrBaseColorTexture;
	float pbrMetallicFactor;
	float pbrRoughnessFactor;
	int   pbrMetallicRoughnessTexture;
	// 8
	vec4 khrDiffuseFactor;  // KHR_materials_pbrSpecularGlossiness
	vec3 khrSpecularFactor;
	int  khrDiffuseTexture;
	// 16
	int   shadingModel;  // 0: metallic-roughness, 1: specular-glossiness
	float khrGlossinessFactor;
	int   khrSpecularGlossinessTexture;
	int   emissiveTexture;
	// 20
	vec3 emissiveFactor;
	int  alphaMode;
	// 24
	float alphaCutoff;
	int   doubleSided;
	int   normalTexture;
	float normalTextureScale;
	// 28
	mat4 uvTransform;
	// 32
	int unlit;

	float transmissionFactor;
	int   transmissionTexture;

	float ior;
	// 36
	vec3  anisotropyDirection;
	float anisotropy;
	// 40
	vec3  attenuationColor;
	float thicknessFactor;  // 44
	int   thicknessTexture;
	float attenuationDistance;
	// --
	float clearcoatFactor;
	float clearcoatRoughness;
	// 48
	int  clearcoatTexture;
	int  clearcoatRoughnessTexture;
	uint sheen;
	int  pad;
	// 52
};

// Use with PushConstant
struct RtxState
{
	int   frame;                  // Current frame, start at 0
	int   maxDepth;               // How deep the path is
	int   spp;
	float fireflyClampThreshold;  // to cut fireflies

	float hdrMultiplier;          // To brightening the scene
	int   debugging_mode;         // See DebugMode
	int   pbrMode;                // 0-Disney, 1-Gltf
	float environmentProb;        // Used in direct light importance sampling

	ivec2 size;                   // rendering size
	int   minHeatmap;             // Debug mode - heat map
	int   maxHeatmap;
	uint time;                   // How long has the app been running. miliseconds.
};

// Structure used for retrieving the primitive information in the closest hit
// using gl_InstanceCustomIndexNV
struct InstanceData
{
	uint64_t vertexAddress;
	uint64_t indexAddress;
	int      materialIndex;
};


// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual

const int LightType_Directional = 0;
const int LightType_Point = 1;
const int LightType_Spot = 2;

// custom light source for direct light importance sampling
const int LightType_Triangle = 3;

// ReSTIR
struct LightSample {
	vec3 Li;
	vec3 wi;
	float dist;
};

struct Reservoir {
	LightSample lightSample;
	uint num;
	float weight;
};

// acceleration structure for importance sampling - pre-computed
struct ImptSampData
{
	int  alias;
	float q;
	float pdf;
	float aliasPdf;
};

struct PuncLight // point, spot, or directional light.
{
	int   type;
	vec3  direction;

	float intensity;
	vec3  color;

	vec3  position;
	float range;

	float outerConeCos;
	float innerConeCos;
	vec2 padding;

	ImptSampData impSamp;
};

struct TrigLight { // triangles of emissive meshes
	uint matIndex;
	uint transformIndex;
	vec3 v0;
	vec3 v1;
	vec3 v2;
	vec2 uv0;
	vec2 uv1;
	vec2 uv2;
	ImptSampData impSamp;
	vec3 pad;
};

struct LightBufInfo {
	uint puncLightSize;
	uint trigLightSize;
	float trigSampProb;
	int pad;
};

// Tonemapper used in post.frag
struct Tonemapper
{
	float brightness;
	float contrast;
	float saturation;
	float vignette;

	float avgLum;
	float zoom;
	vec2  renderingRatio;

	int   autoExposure;
	float Ywhite;  // Burning white
	float key;     // Log-average luminance
	int pad;
};


struct SunAndSky
{
	vec3  rgb_unit_conversion;
	float multiplier;

	float haze;
	float redblueshift;
	float saturation;
	float horizon_height;

	vec3  ground_color;
	float horizon_blur;

	vec3  night_color;
	float sun_disk_intensity;

	vec3  sun_direction;
	float sun_disk_scale;

	float sun_glow_intensity;
	int   y_is_up;
	int   physically_scaled_sun;
	int   in_use;
};


#endif  // COMMON_HOST_DEVICE
