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
// This file holds the layout used by all ray tracing shaders


#ifndef LAYOUTS_GLSL
#define LAYOUTS_GLSL 1


// C++ shared structures and binding
#include "host_device.h"

//----------------------------------------------
// Descriptor Set Layout
//----------------------------------------------


// clang-format off
layout(set = S_ACCEL, binding = eTlas)					uniform accelerationStructureEXT topLevelAS;
//
layout(set = S_OUT,   binding = eLastDirectResult)   uniform readonly image2D lastDirectResultImage;
layout(set = S_OUT,   binding = eLastIndirectResult) uniform readonly image2D lastIndirectResultImage;

layout(set = S_OUT,   binding = eThisDirectResult)   uniform image2D thisDirectResultImage;
layout(set = S_OUT,   binding = eThisIndirectResult) uniform image2D thisIndirectResultImage;
//
layout(set = S_SCENE, binding = eInstData,	scalar)   buffer _InstanceInfo	{ InstanceData geoInfo[]; };
layout(set = S_SCENE, binding = eCamera,	  scalar)   uniform _SceneCamera	{ SceneCamera sceneCamera; };
layout(set = S_SCENE, binding = eMaterials,	scalar)		buffer _MaterialBuffer	{ GltfShadeMaterial materials[]; };
layout(set = S_SCENE, binding = ePuncLights,scalar)		buffer _PuncLights		{ PuncLight puncLights[]; };
layout(set = S_SCENE, binding = eTrigLights,scalar)		buffer _TrigLights		{ TrigLight trigLights[]; };
layout(set = S_SCENE, binding = eLightBufInfo     )		uniform _LightBufInfo		{ LightBufInfo lightBufInfo; };
layout(set = S_SCENE, binding = eTextures         )   uniform sampler2D		texturesMap[]; 
//
layout(set = S_ENV, binding = eSunSky,		scalar)		uniform _SSBuffer		{ SunAndSky _sunAndSky; };
layout(set = S_ENV, binding = eHdr)						uniform sampler2D		environmentTexture;
layout(set = S_ENV, binding = eImpSamples,  scalar)		buffer _EnvSampBuffer	{ ImptSampData envSamplingData[]; };

layout(set = S_RAYQ, binding = eLastGbuffer)       uniform readonly uimage2D lastGbuffer; 
layout(set = S_RAYQ, binding = eThisGbuffer)       uniform uimage2D thisGbuffer;
layout(set = S_RAYQ, binding = eMotionVector)      uniform iimage2D motionVector;

layout(set = S_RAYQ, binding = eLastDirectResv, scalar) buffer _LastDirectResv { DirectReservoir lastDirectResv[]; };
layout(set = S_RAYQ, binding = eThisDirectResv, scalar) buffer _ThisDirectResv { DirectReservoir thisDirectResv[]; };
layout(set = S_RAYQ, binding = eTempDirectResv, scalar) buffer _TempDirectResv { DirectReservoir tempDirectResv[]; };

layout(set = S_RAYQ, binding = eLastIndirectResv, scalar) buffer _LastIndirectResv { IndirectReservoir lastIndirectResv[]; };
layout(set = S_RAYQ, binding = eThisIndirectResv, scalar) buffer _ThisIndirectResv { IndirectReservoir thisIndirectResv[]; };
layout(set = S_RAYQ, binding = eTempIndirectResv, scalar) buffer _TempIndirectResv { IndirectReservoir tempIndirectResv[]; };

layout(set = S_RAYQ, binding = eDenoiseDirTempA) uniform image2D denoiseDirTempA;
layout(set = S_RAYQ, binding = eDenoiseDirTempB) uniform image2D denoiseDirTempB;
layout(set = S_RAYQ, binding = eDenoiseIndTempA) uniform image2D denoiseIndTempA;
layout(set = S_RAYQ, binding = eDenoiseIndTempB) uniform image2D denoiseIndTempB;

layout(buffer_reference, scalar) buffer Vertices { VertexAttributes v[]; };
layout(buffer_reference, scalar) buffer Indices	 { uvec3 i[];            };

  // clang-format on


#endif  // LAYOUTS_GLSL
