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

//--------------------------------------------------------------------------------------------------
// This creates the image in floating point, holding the result of ray tracing.
// It also creates a pipeline for drawing this image from HDR to LDR applying a tonemapper
//


#pragma once

#include "nvmath/nvmath.h"

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "shaders/host_device.h"

#include <array>

class RenderOutput
{
public:
  struct PushConstant {
      Tonemapper tm;
      int debugging_mode;
  } m_push;
  Tonemapper m_tm{
      1.0f,          // brightness;
      1.0f,          // contrast;
      1.0f,          // saturation;
      0.0f,          // vignette;
      1.0f,          // avgLum;
      1.0f,          // zoom;
      {1.0f, 1.0f},  // renderingRatio;
      0,             // autoExposure;
      0.5f,          // Ywhite;  // Burning white
      0.5f,          // key;     // Log-average luminance
  };
  Tonemapper m_depthTm{
        0.0f,          // exposure;
        2.2f,          // gamma;
        0.0f,          // alpha;
  };

public:
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount);
  void destroy();
  void create(const VkExtent2D& size, const VkRenderPass& renderPass);
  void update(const VkExtent2D& size);
  void run(VkCommandBuffer cmdBuf, const RtxState& state, float zoom, vec2 ratio, int frames);
  void genMipmap(VkCommandBuffer cmdBuf);

  VkDescriptorSetLayout getDescLayout() { return m_postDescSetLayout; }
  VkDescriptorSet getDescSet(int frames) { return m_postDescSet[(frames + 1) % 2]; }

private:
  void createOffscreenRender(const VkExtent2D& size);
  void createPostPipeline(const VkRenderPass& renderPass);
  void createPostDescriptor();

  uint32_t m_imageCount;

  nvvk::DescriptorSetBindings m_bind;
  VkDescriptorPool      m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_postDescSetLayout{VK_NULL_HANDLE};
  std::array<VkDescriptorSet, 2> m_postDescSet{VK_NULL_HANDLE};
  VkPipeline            m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout      m_postPipelineLayout{VK_NULL_HANDLE};
  //nvvk::Texture         m_offscreenColor;
  std::array<nvvk::Texture, 2>         m_directResult;
  std::array<nvvk::Texture, 2>         m_indirectResult;
  //VkFormat m_offscreenColorFormat{VkFormat::eR16G16B16A16Sfloat};  // Darkening the scene over 5000 iterations

  // Direct: RGB color and light source distance
  // Indirect: RGB color and screen depth
  VkFormat m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};  // Will be replaced by best supported format


  // Setup
  nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;   // Utility to name objects
  VkDevice                 m_device;
  uint32_t                 m_queueIndex;

  VkExtent2D m_size{};
};
