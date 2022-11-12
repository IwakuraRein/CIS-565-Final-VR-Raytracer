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


#pragma once

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvmath/nvmath.h"

#include "shaders/host_device.h"
#include "scene.hpp"

#include <array>

/*

Creating the Compute ray query renderer 
* Requiring:  
  - Acceleration structure (AccelSctruct / Tlas)
  - An image (Post StoreImage)
  - The glTF scene (vertex, index, materials, ... )

* Usage
  - setup as usual
  - create
  - run
*/
class Renderer
{
public:
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount);
  void destroy();
  void create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout> rtDescSetLayouts, Scene* scene);
  void              run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets);
  const std::string name() { return std::string("RQ"); }
  void update(const VkExtent2D& size);
  void createDescriptorSet();
  void setPushContants(const RtxState& state) { m_state = state; }

private:
  uint32_t m_nbHit{0};
  uint32_t m_imageCount;

private:
  // Setup

  RtxState m_state{};
  nvvk::ResourceAllocator* m_pAlloc{nullptr};  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;            // Utility to name objects
  VkDevice                 m_device{VK_NULL_HANDLE};
  uint32_t                 m_queueIndex{0};

  std::array<nvvk::Buffer, 2> m_buffer;
  uint m_bufferSize;
  nvvk::DescriptorSetBindings m_bind;
  std::array<VkDescriptorBufferInfo, 2> m_dbi;
  VkDescriptorPool      m_descPool{ VK_NULL_HANDLE };
  VkDescriptorSetLayout m_descSetLayout{ VK_NULL_HANDLE };
  std::array<VkDescriptorSet, 2> m_descSet{ VK_NULL_HANDLE };

  VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
  VkPipeline       m_pipeline{VK_NULL_HANDLE};
};
