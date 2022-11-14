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
// This is called by the post process shader to display the result of ray tracing.
// It applied a tonemapper and do dithering on the image to avoid banding.

#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_debug_printf : enable
#extension GL_ARB_gpu_shader_int64 : enable  // Shader reference

#define TONEMAP_UNCHARTED
#include "random.glsl"
#include "tonemapping.glsl"
#include "host_device.h"

layout(location = 0) in vec2 uvCoords;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D inDirectImage;
layout(set = 0, binding = 1) uniform sampler2D inIndirectImage;

layout(push_constant) uniform _PushConstant {
  Tonemapper tm;
  float zoom;
  vec2 renderingRatio;
  int debugging_mode;
};

vec3 getColor(){
  // vec3 direct = imageLoad(thisDirectResultImage, ivec2(gl_FragCoord.x, gl_FragCoord.y)).xyz;
  // vec3 indirect = imageLoad(thisIndirectResultImage, ivec2(gl_FragCoord.x, gl_FragCoord.y)).xyz;
  vec3 direct = texture(inDirectImage, uvCoords * zoom).rgb;
  vec3 indirect = texture(inIndirectImage, uvCoords * zoom).rgb;
  if (debugging_mode == eNoDebug)
    return direct + indirect;
  if (debugging_mode == eIndirectStage)
    return indirect;
  return direct;
}

void main() {
  vec3 color = getColor();
  if (debugging_mode != eNormal &&
      debugging_mode != eRoughness &&
      debugging_mode != eMetallic &&
      debugging_mode != eAlpha &&
      debugging_mode != eTexcoord &&
      debugging_mode != eTangent
  ){
    color *= pow(2, tm.exposure);
    const float W   = 11.2;
    color           = toneMapUncharted2Impl(color * 2.0);
    vec3 whiteScale = 1.0 / toneMapUncharted2Impl(vec3(W));
    color *= whiteScale;
    color += tm.alpha;
    color = clamp(pow(color, vec3(1.f / tm.gamma)), 0.f, 1.f);
  }
  fragColor.xyz = color;
  fragColor.w = 1.0;
}
