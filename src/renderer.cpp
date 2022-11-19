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
#include "autogen/indirect_stage1.comp.h"
#include "autogen/indirect_stage2.comp.h"
#include "autogen/indirect_stage3.comp.h"
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
	m_pAlloc->destroy(m_reservoir[0]);
	m_pAlloc->destroy(m_reservoir[1]);
	m_pAlloc->destroy(m_radianceCache[0]);
	m_pAlloc->destroy(m_radianceCache[1]);
	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	vkDestroyPipeline(m_device, m_directPipeline, nullptr);
	vkDestroyPipeline(m_device, m_indirectPipeline1, nullptr);
	vkDestroyPipeline(m_device, m_indirectPipeline2, nullptr);
	vkDestroyPipeline(m_device, m_indirectPipeline3, nullptr);
	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

	m_pipelineLayout = VK_NULL_HANDLE;
	m_directPipeline = VK_NULL_HANDLE;
	m_indirectPipeline1 = VK_NULL_HANDLE;
	m_indirectPipeline2 = VK_NULL_HANDLE;
	m_indirectPipeline3 = VK_NULL_HANDLE;
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
	createBuffer();

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


	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, indirect_stage1_comp, sizeof(indirect_stage1_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_indirectPipeline1);
	m_debug.setObjectName(m_indirectPipeline1, "RendererIndirect1");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, indirect_stage2_comp, sizeof(indirect_stage2_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_indirectPipeline2);
	m_debug.setObjectName(m_indirectPipeline2, "RendererIndirect2");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, indirect_stage3_comp, sizeof(indirect_stage3_comp));
	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_indirectPipeline3);
	m_debug.setObjectName(m_indirectPipeline3, "RendererIndirect3");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);

	timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
#define GROUP_SIZE 8  // Same group size as in compute shader
void Renderer::run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets, int frames)
{
	// Preparing for the compute shader
	descSets.push_back(m_descSet[(frames + 1) % 2]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directPipeline);
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Sending the push constant information
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &state);

	// Dispatching the shader
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

	// Create Radiance Cache
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline1);
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

	// Path trace radiance cache
	size_t size = (m_size.width + GROUP_SIZE - 1) / GROUP_SIZE;
	size *= (m_size.height + GROUP_SIZE - 1) / GROUP_SIZE;
	size *= CACHES_PER_GROUP;
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline2);
	vkCmdDispatch(cmdBuf, (size + (GROUP_SIZE - 1)) / GROUP_SIZE, 1, 1);

	// Shade from radiance cache
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline3);
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

// handle window resize
void Renderer::update(const VkExtent2D& size) {
	//if ((size.width * size.height) > (m_size.width * m_size.height)) {
		m_size = size;
		createImage();
		createBuffer();

		VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
			| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorBufferInfo, 2> dbi;
		dbi[0] = VkDescriptorBufferInfo{ m_radianceCache[0].buffer, 0, VK_WHOLE_SIZE };
		dbi[1] = VkDescriptorBufferInfo{ m_radianceCache[1].buffer, 0, VK_WHOLE_SIZE };
		std::array<VkWriteDescriptorSet, 6> writes;
		writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_gbuffer[0].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_gbuffer[1].descriptor);
		writes[2] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastDirectCache, &m_reservoir[0].descriptor);
		writes[3] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisDirectCache, &m_reservoir[1].descriptor);
		writes[4] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastIndirectCache, &dbi[0]);
		writes[5] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisIndirectCache, &dbi[1]);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_gbuffer[1].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_gbuffer[0].descriptor);
		writes[2] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastDirectCache, &m_reservoir[1].descriptor);
		writes[3] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisDirectCache, &m_reservoir[0].descriptor);
		writes[4] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastIndirectCache, &dbi[1]);
		writes[5] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisIndirectCache, &dbi[0]);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	//}
}
void Renderer::createBuffer() 
{ // create radiance cache buffer
	size_t size = (m_size.width + GROUP_SIZE - 1) / GROUP_SIZE;
	size *= (m_size.height + GROUP_SIZE - 1) / GROUP_SIZE;
	size *= CACHES_PER_GROUP;

	m_pAlloc->destroy(m_radianceCache[0]);
	m_pAlloc->destroy(m_radianceCache[1]);
	m_radianceCache[0] = m_pAlloc->createBuffer(sizeof(RadianceCacheStorage) * size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	NAME_VK(m_radianceCache[0].buffer);
	m_radianceCache[1] = m_pAlloc->createBuffer(sizeof(RadianceCacheStorage) * size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	NAME_VK(m_radianceCache[1].buffer);
}

void Renderer::createImage()
{  // Creating the color image
	m_pAlloc->destroy(m_gbuffer[0]);
	m_pAlloc->destroy(m_gbuffer[1]);
	m_pAlloc->destroy(m_reservoir[0]);
	m_pAlloc->destroy(m_reservoir[1]);
	{
		// gbuffer
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
		
		// reservoir
		colorCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_reservoirFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);


		nvvk::Image cacheimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage1.image);
		nvvk::Image cacheimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(cacheimage2.image);
		ivInfo1 = nvvk::makeImageViewCreateInfo(cacheimage1.image, colorCreateInfo);
		ivInfo2 = nvvk::makeImageViewCreateInfo(cacheimage2.image, colorCreateInfo);
		m_reservoir[0] = m_pAlloc->createTexture(cacheimage1, ivInfo1);
		m_reservoir[1] = m_pAlloc->createTexture(cacheimage2, ivInfo2);

		m_reservoir[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_reservoir[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_reservoir[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_reservoir[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		
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
	m_bind.addBinding({ RayQBindings::eLastIndirectCache, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisIndirectCache, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size());
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	

	std::array<VkDescriptorBufferInfo, 2> dbi;
	dbi[0] = VkDescriptorBufferInfo{ m_radianceCache[0].buffer, 0, VK_WHOLE_SIZE };
	dbi[1] = VkDescriptorBufferInfo{ m_radianceCache[1].buffer, 0, VK_WHOLE_SIZE };
	std::array<VkWriteDescriptorSet, 6> writes;
	writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_gbuffer[0].descriptor);
	writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_gbuffer[1].descriptor);
	writes[2] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastDirectCache, &m_reservoir[0].descriptor);
	writes[3] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisDirectCache, &m_reservoir[1].descriptor);
	writes[4] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastIndirectCache, &dbi[0]);
	writes[5] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisIndirectCache, &dbi[1]);

	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_gbuffer[1].descriptor);
	writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_gbuffer[0].descriptor);
	writes[2] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastDirectCache, &m_reservoir[1].descriptor);
	writes[3] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisDirectCache, &m_reservoir[0].descriptor);
	writes[4] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastIndirectCache, &dbi[1]);
	writes[5] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisIndirectCache, &dbi[0]);
	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
