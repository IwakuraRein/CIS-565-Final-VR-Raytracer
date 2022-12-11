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
 *  This creates the image in floating point, holding the result of ray tracing.
 *  It also creates a pipeline for drawing this image from HDR to LDR applying a tonemapper
 */


#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "render_output.hpp"
#include "tools.hpp"

// Shaders
#include "autogen/passthrough.vert.h"
#include "autogen/post.frag.h"


void RenderOutput::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount)
{
  m_device     = device;
  m_imageCount = imageCount;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);

  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}


void RenderOutput::destroy()
{
  for (int i = 0; i < 2; i++) {
      m_pAlloc->destroy(m_directResult[i]);
      m_pAlloc->destroy(m_indirectResult[i]);
  }

  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
}

void RenderOutput::create(const VkExtent2D& size, const VkRenderPass& renderPass)
{
  MilliTimer timer;
  LOGI("Create Offscreen");
  createOffscreenRender(size);
  createPostPipeline(renderPass);
  timer.print();
}

void RenderOutput::update(const VkExtent2D& size)
{
  createOffscreenRender(size);
}

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void RenderOutput::createOffscreenRender(const VkExtent2D& size)
{
  m_size = size;
  if(m_directResult[0].image != VK_NULL_HANDLE)
  {
    m_pAlloc->destroy(m_directResult[0]);
  }
  if(m_directResult[1].image != VK_NULL_HANDLE)
  {
    m_pAlloc->destroy(m_directResult[1]);
  }
  if(m_indirectResult[0].image != VK_NULL_HANDLE)
  {
    m_pAlloc->destroy(m_indirectResult[0]);
  }
  if(m_indirectResult[1].image != VK_NULL_HANDLE)
  {
    m_pAlloc->destroy(m_indirectResult[1]);
  }

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
        size, m_offscreenColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true
    );

    nvvk::Image directImage1 = m_pAlloc->createImage(colorCreateInfo);
    NAME_VK(directImage1.image);
    nvvk::Image directImage2 = m_pAlloc->createImage(colorCreateInfo);
    NAME_VK(directImage2.image);
    VkImageViewCreateInfo ivInfo1 = nvvk::makeImageViewCreateInfo(directImage1.image, colorCreateInfo);
    VkImageViewCreateInfo ivInfo2 = nvvk::makeImageViewCreateInfo(directImage2.image, colorCreateInfo);

    nvvk::Image inDirectImage1 = m_pAlloc->createImage(colorCreateInfo);
    NAME_VK(inDirectImage1.image);
    nvvk::Image inDirectImage2 = m_pAlloc->createImage(colorCreateInfo);
    NAME_VK(inDirectImage2.image);
    VkImageViewCreateInfo ivInfo3 = nvvk::makeImageViewCreateInfo(inDirectImage1.image, colorCreateInfo);
    VkImageViewCreateInfo ivInfo4 = nvvk::makeImageViewCreateInfo(inDirectImage2.image, colorCreateInfo);

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.maxLod                             = FLT_MAX;
    m_directResult[0]                          = m_pAlloc->createTexture(directImage1, ivInfo1, sampler);
    m_directResult[1]                          = m_pAlloc->createTexture(directImage2, ivInfo2, sampler);
    m_indirectResult[0]                        = m_pAlloc->createTexture(inDirectImage1, ivInfo3, sampler);
    m_indirectResult[1]                        = m_pAlloc->createTexture(inDirectImage2, ivInfo4, sampler);
    m_directResult[0].descriptor.imageLayout   = VK_IMAGE_LAYOUT_GENERAL;
    m_directResult[1].descriptor.imageLayout   = VK_IMAGE_LAYOUT_GENERAL;
    m_indirectResult[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_indirectResult[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_directResult[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectResult[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_directResult[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectResult[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  createPostDescriptor();
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void RenderOutput::createPostPipeline(const VkRenderPass& renderPass)
{
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);

  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.setLayoutCount         = 1;
  pipelineLayoutCreateInfo.pSetLayouts            = &m_postDescSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_postPipelineLayout);

  // Pipeline: completely generic, no vertices
  std::vector<uint32_t> vertexShader(std::begin(passthrough_vert), std::end(passthrough_vert));
  std::vector<uint32_t> fragShader(std::begin(post_frag), std::end(post_frag));

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, renderPass);
  pipelineGenerator.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  CREATE_NAMED_VK(m_postPipeline, pipelineGenerator.createPipeline());
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void RenderOutput::createPostDescriptor()
{
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);

  m_bind = nvvk::DescriptorSetBindings{};
  // This descriptor is passed to the RTX pipeline
  // Ray tracing will write to the binding 1, but the fragment shader will be using binding 0, so it can use a sampler too.
  m_bind.addBinding({OutputBindings::eDirectSampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT});
  m_bind.addBinding({OutputBindings::eIndirectSampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT});
  
  m_bind.addBinding({OutputBindings::eThisDirectResult, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR});
  m_bind.addBinding({OutputBindings::eThisIndirectResult, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR});
  m_bind.addBinding({OutputBindings::eLastDirectResult, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR});
  m_bind.addBinding({OutputBindings::eLastIndirectResult, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR});
  m_postDescSetLayout = m_bind.createLayout(m_device);
  m_postDescPool      = m_bind.createPool(m_device, m_postDescSet.size());
  m_postDescSet[0]    = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
  m_postDescSet[1]    = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);

  std::vector<VkWriteDescriptorSet> writes;
  for (int i = 0; i < 2; i++) {
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eDirectSampler, &m_directResult[i].descriptor));
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eIndirectSampler, &m_indirectResult[i].descriptor));
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eThisDirectResult, &m_directResult[i].descriptor));
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eLastDirectResult, &m_directResult[!i].descriptor));
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eThisIndirectResult, &m_indirectResult[i].descriptor));
      writes.emplace_back(m_bind.makeWrite(m_postDescSet[i], OutputBindings::eLastIndirectResult, &m_indirectResult[!i].descriptor));

      vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void RenderOutput::run(VkCommandBuffer cmdBuf, const RtxState& state, float zoom, vec2 ratio, int frames)
{
  LABEL_SCOPE_VK(cmdBuf);

  m_push.debugging_mode = state.debugging_mode;
  if (state.debugging_mode == eDepth) m_push.tm = m_depthTm;
  else m_push.tm = m_tm;
  m_push.tm.zoom = zoom;
  m_push.tm.renderingRatio = ratio;
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &m_push);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet[(frames + 1) % 2], 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}

//--------------------------------------------------------------------------------------------------
// Generating all pyramid images, the highest level is used for getting the average luminance
// of the image, which is then use to auto-expose.
//
void RenderOutput::genMipmap(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  nvvk::cmdGenerateMipmaps(cmdBuf, m_directResult[0].image, m_offscreenColorFormat, m_size, nvvk::mipLevels(m_size), 1,
      VK_IMAGE_LAYOUT_GENERAL);
  nvvk::cmdGenerateMipmaps(cmdBuf, m_directResult[1].image, m_offscreenColorFormat, m_size, nvvk::mipLevels(m_size), 1,
      VK_IMAGE_LAYOUT_GENERAL);
  nvvk::cmdGenerateMipmaps(cmdBuf, m_indirectResult[0].image, m_offscreenColorFormat, m_size, nvvk::mipLevels(m_size), 1,
      VK_IMAGE_LAYOUT_GENERAL);
  nvvk::cmdGenerateMipmaps(cmdBuf, m_indirectResult[1].image, m_offscreenColorFormat, m_size, nvvk::mipLevels(m_size), 1,
      VK_IMAGE_LAYOUT_GENERAL);
}
