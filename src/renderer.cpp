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
  *  Implement ray tracing using Ray-Query in a compute shader.
  *  This allows to compare the speed with RTX, but also to easier debug the shading pipeline,
  *  as it is not building a Mega kernel as the RTX pipeline does.
  */



#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "renderer.hpp"
#include "tools.hpp"

  // Shaders
#include "autogen/direct_stage.comp.h"
#include "autogen/indirect_stage.comp.h"
//--------------------------------------------------------------------------------------------------
//
//
void Renderer::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount)
{
	m_device = device;
	m_imageCount = imageCount;
	m_pAlloc = allocator;
	m_queueIndex = familyIndex;
	m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
//
//
void Renderer::destroy()
{
	m_pAlloc->destroy(m_gbuffer[0]);
	m_pAlloc->destroy(m_gbuffer[1]);
	m_pAlloc->destroy(m_directCache[0]);
	m_pAlloc->destroy(m_directCache[1]);
	m_pAlloc->destroy(m_indirectCache[0]);
	m_pAlloc->destroy(m_indirectCache[1]);
	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	vkDestroyPipeline(m_device, m_directPipeline, nullptr);
	vkDestroyPipeline(m_device, m_indirectPipeline, nullptr);
	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

	m_pipelineLayout = VK_NULL_HANDLE;
	m_directPipeline = VK_NULL_HANDLE;
	m_indirectPipeline = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Creation of the RQ pipeline
//
void Renderer::create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout> rtDescSetLayouts, Scene* scene)
{
	m_size = size;
	MilliTimer timer;
	LOGI("Create Ray Query Pipeline");

	std::vector<VkPushConstantRange> push_constants;
	push_constants.push_back({ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState) });

	// Create Gbuffer
	createImage();

	createDescriptorSet();
	rtDescSetLayouts.push_back(m_descSetLayout);

	VkPipelineLayoutCreateInfo layout_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constants.size());
	layout_info.pPushConstantRanges = push_constants.data();
	layout_info.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
	layout_info.pSetLayouts = rtDescSetLayouts.data();
	vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipelineLayout);

	VkComputePipelineCreateInfo computePipelineCreateInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	computePipelineCreateInfo.layout = m_pipelineLayout;
	computePipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, direct_stage_comp, sizeof(direct_stage_comp));
	computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computePipelineCreateInfo.stage.pName = "main";

	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_directPipeline);

	m_debug.setObjectName(m_directPipeline, "Renderer-Direct");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);
	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, indirect_stage_comp, sizeof(indirect_stage_comp));

	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_indirectPipeline);

	m_debug.setObjectName(m_indirectPipeline, "RendererIndirect");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
#define GROUP_SIZE 8  // Same group size as in compute shader
void Renderer::run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets)
{

	// Preparing for the compute shader
	descSets.push_back(m_descSet[(state.frame + 1) % 2]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directPipeline);
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Sending the push constant information
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &state);

	// Dispatching the shader
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline);
	vkCmdDispatch(cmdBuf, (state.size[0] / state.descale + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] / state.descale + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

// handle window resize
void Renderer::update(const VkExtent2D& size) {
	//if ((size.width * size.height) > (m_size.width * m_size.height)) {
		m_size = size;
		m_pAlloc->destroy(m_gbuffer[0]);
		m_pAlloc->destroy(m_gbuffer[1]);
		m_pAlloc->destroy(m_directCache[0]);
		m_pAlloc->destroy(m_directCache[1]);
		m_pAlloc->destroy(m_indirectCache[0]);
		m_pAlloc->destroy(m_indirectCache[1]);
		createImage();

		VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
			| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;


		std::array<VkWriteDescriptorSet, 6> writes;
		writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_gbuffer[0].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_gbuffer[1].descriptor);
		writes[2] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastDirectCache, &m_directCache[0].descriptor);
		writes[3] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisDirectCache, &m_directCache[1].descriptor);
		writes[4] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastIndirectCache, &m_indirectCache[0].descriptor);
		writes[5] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisIndirectCache, &m_indirectCache[1].descriptor);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_gbuffer[1].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_gbuffer[0].descriptor);
		writes[2] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastDirectCache, &m_directCache[1].descriptor);
		writes[3] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisDirectCache, &m_directCache[0].descriptor);
		writes[4] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastIndirectCache, &m_indirectCache[1].descriptor);
		writes[5] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisIndirectCache, &m_indirectCache[0].descriptor);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	//}
}

void Renderer::createImage()
{  // Creating the color image
	{
		auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_gbufferFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		nvvk::Image gbimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage1.image);
		nvvk::Image gbimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage2.image);
		VkImageViewCreateInfo ivInfo1 = nvvk::makeImageViewCreateInfo(gbimage1.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo2 = nvvk::makeImageViewCreateInfo(gbimage2.image, colorCreateInfo);

		m_gbuffer[0] = m_pAlloc->createTexture(gbimage1, ivInfo1);
		m_gbuffer[1] = m_pAlloc->createTexture(gbimage2, ivInfo2);
		m_gbuffer[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_gbuffer[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		nvvk::Image cacheimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage1.image);
		nvvk::Image cacheimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage2.image);
		nvvk::Image cacheimage3 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage3.image);
		nvvk::Image cacheimage4 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage4.image);
		ivInfo1 = nvvk::makeImageViewCreateInfo(cacheimage1.image, colorCreateInfo);
		ivInfo2 = nvvk::makeImageViewCreateInfo(cacheimage2.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo3 = nvvk::makeImageViewCreateInfo(cacheimage3.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo4 = nvvk::makeImageViewCreateInfo(cacheimage4.image, colorCreateInfo);

		m_directCache[0] = m_pAlloc->createTexture(cacheimage1, ivInfo1);
		m_directCache[1] = m_pAlloc->createTexture(cacheimage2, ivInfo2);
		m_indirectCache[0] = m_pAlloc->createTexture(cacheimage3, ivInfo3);
		m_indirectCache[1] = m_pAlloc->createTexture(cacheimage4, ivInfo4);
		m_directCache[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_directCache[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_indirectCache[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_indirectCache[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_directCache[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_directCache[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectCache[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_indirectCache[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		
		genCmdBuf.submitAndWait(cmdBuf);
	}
}

void Renderer::createDescriptorSet()
{
	//if (m_descPool != VK_NULL_HANDLE)
	//	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	//if (m_descSetLayout != VK_NULL_HANDLE)
	//	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
	m_bind = nvvk::DescriptorSetBindings{};

	VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	m_bind.addBinding({ RayQBindings::eLastGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eLastDirectCache, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisDirectCache, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eLastIndirectCache, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisIndirectCache, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size());
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	std::array<VkWriteDescriptorSet, 6> writes;
	writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_gbuffer[0].descriptor);
	writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_gbuffer[1].descriptor);
	writes[2] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastDirectCache, &m_directCache[0].descriptor);
	writes[3] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisDirectCache, &m_directCache[1].descriptor);
	writes[4] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastIndirectCache, &m_indirectCache[0].descriptor);
	writes[5] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisIndirectCache, &m_indirectCache[1].descriptor);
	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_gbuffer[1].descriptor);
	writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_gbuffer[0].descriptor);
	writes[2] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastDirectCache, &m_directCache[1].descriptor);
	writes[3] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisDirectCache, &m_directCache[0].descriptor);
	writes[4] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastIndirectCache, &m_indirectCache[1].descriptor);
	writes[5] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisIndirectCache, &m_indirectCache[0].descriptor);
	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
