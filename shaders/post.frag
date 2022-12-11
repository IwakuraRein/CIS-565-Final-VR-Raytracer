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
#include "compress.glsl"
#include "tonemapping.glsl"
#include "host_device.h"

layout(location = 0) in vec2 uvCoords;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D inDirectImage;
layout(set = 0, binding = 1) uniform sampler2D inIndirectImage;

layout(push_constant) uniform _PushConstant {
  Tonemapper tm;
  int debugging_mode;
};

vec2 indCoord;

// http://www.thetenthplanet.de/archives/5367
// Apply dithering to hide banding artifacts.
vec3 dither(vec3 linear_color, vec3 noise, float quant) {
  vec3 c0 = floor(linearTosRGB(linear_color) / quant) * quant;
  vec3 c1 = c0 + quant;
  vec3 discr = mix(sRGBToLinear(c0), sRGBToLinear(c1), noise);
  return mix(c0, c1, lessThan(discr, linear_color));
}

// http://user.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
const mat3 RGB2XYZ = mat3(0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041);
float luminance(vec3 color) {
  return dot(color, vec3(0.2126f, 0.7152f, 0.0722f));  //color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
}

vec3 toneExposure(vec3 RGB, float logAvgLum) {
  vec3 XYZ = RGB2XYZ * RGB;
  float Y = (tm.key / logAvgLum) * XYZ.y;
  float Yd = (Y * (1.0 + Y / (tm.Ywhite * tm.Ywhite))) / (1.0 + Y);
  return RGB / XYZ.y * Yd;
}

vec3 toneLocalExposure(vec3 RGB, float logAvgLum) {
  vec3 XYZ = RGB2XYZ * RGB;
  float Y = (tm.key / logAvgLum) * XYZ.y;
  float La;  // local adaptation luminance
  float factor = tm.key / logAvgLum;
  float epsilon = 0.05, phi = 2.0;
  float scale[7] = float[7](1, 2, 4, 8, 16, 32, 64);
  for(int i = 0; i < 7; ++i) {
    float v1;
    if(debugging_mode == eDirectStage)
      v1 = luminance(texture(inDirectImage, uvCoords * tm.zoom, i).rgb) * factor;
    else if(debugging_mode == eIndirectStage)
      v1 = luminance(texture(inIndirectImage, indCoord * tm.zoom, i).rgb) * factor;
    else
      v1 = luminance(texture(inDirectImage, uvCoords * tm.zoom, i).rgb + texture(inIndirectImage, indCoord * tm.zoom, i).rgb) * factor;
    float v2;
    if(debugging_mode == eDirectStage)
      v2 = luminance(texture(inDirectImage, uvCoords * tm.zoom, i + 1).rgb) * factor;
    else if(debugging_mode == eIndirectStage)
      v2 = luminance(texture(inIndirectImage, indCoord * tm.zoom, i + 1).rgb) * factor;
    else
      v2 == luminance(texture(inDirectImage, uvCoords * tm.zoom, i + 1).rgb + texture(inDirectImage, indCoord * tm.zoom, i + 1).rgb) * factor;
    if(abs(v1 - v2) / ((tm.key * pow(2, phi) / (scale[i] * scale[i])) + v1) > epsilon) {
      La = v1;
      break;
    } else
      La = v2;
  }
  float Yd = Y / (1.0 + La);

  return RGB / XYZ.y * Yd;
}

void main() {
  indCoord = uvCoords;

  if (debugging_mode == eDepth){
    float depth = texture(inDirectImage, uvCoords * tm.zoom).w;
    depth *= pow(2, tm.brightness);
    depth += tm.saturation;
    depth = clamp(pow(depth, 1.0 / tm.contrast), 0.f, 1.f);
    fragColor = vec4(depth, depth, depth, 1.0);
  }
  else if(debugging_mode > eIndirectStage) {
    vec3 color = texture(inDirectImage, indCoord * tm.zoom).xyz;
    if (debugging_mode == eBaseColor)
      color = clamp(pow(color, vec3(0.45454545454545)), 0, 1);
    fragColor = vec4(color, 1.0);
  }
  else {
    // Raw result of ray tracing
    vec4 hdr;
    if(debugging_mode == eDirectStage)
      hdr = texture(inDirectImage, uvCoords * tm.zoom).rgba;
    else if(debugging_mode == eIndirectStage)
      hdr = texture(inIndirectImage, indCoord * tm.zoom).rgba;
    else
      hdr = texture(inDirectImage, uvCoords * tm.zoom).rgba + texture(inIndirectImage, indCoord * tm.zoom).rgba;

    hdr.w = 1.0;
    if(((tm.autoExposure >> 0) & 1) == 1) {
      vec4 avg; // Get the average value of the image
      if(debugging_mode == eDirectStage)
        avg = textureLod(inDirectImage, vec2(0.5), 20);
      else if(debugging_mode == eIndirectStage)
        avg = textureLod(inIndirectImage, vec2(0.5), 20);
      else
        avg = (textureLod(inDirectImage, vec2(0.5), 20) + textureLod(inIndirectImage, vec2(0.5), 20));
      avg.w = 1.0;
      float avgLum2 = luminance(avg.rgb);                  // Find the luminance
      if(((tm.autoExposure >> 1) & 1) == 1)
        hdr.rgb = toneLocalExposure(hdr.rgb, avgLum2);  // Adjust exposure
      else
        hdr.rgb = toneExposure(hdr.rgb, avgLum2);  // Adjust exposure
    }

    // Tonemap + Linear to sRgb
    vec3 color = toneMap(hdr.rgb, tm.avgLum);

    // Remove banding
    uvec3 r = pcg3d(uvec3(gl_FragCoord.xy, 0));
    vec3 noise = uintBitsToFloat(0x3f800000 | (r >> 9)) - 1.0f;
    color = dither(sRGBToLinear(color), noise, 1. / 255.);

    //contrast
    color = clamp(mix(vec3(0.5), color, tm.contrast), 0, 1);
    // brighness
    color = pow(color, vec3(1.0 / tm.brightness));
    // saturation
    vec3 i = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    color = mix(i, color, tm.saturation);
    // vignette
    vec2 uv = ((uvCoords * tm.renderingRatio) - 0.5) * 2.0;
    color *= 1.0 - dot(uv, uv) * tm.vignette;

    fragColor.xyz = color;
    fragColor.w = 1.0;
  }
}
