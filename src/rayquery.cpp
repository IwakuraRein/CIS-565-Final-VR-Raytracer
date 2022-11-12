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
#include "rayquery.hpp"
#include "scene.hpp"
#include "tools.hpp"

  // Shaders
#include "autogen/pathtrace.comp.h"
//--------------------------------------------------------------------------------------------------
//
//
void RayQuery::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount)
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
void RayQuery::destroy()
{
	m_pAlloc->destroy(m_buffer[0]);
	m_pAlloc->destroy(m_buffer[1]);
	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	vkDestroyPipeline(m_device, m_pipeline, nullptr);
	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

	m_pipelineLayout = VK_NULL_HANDLE;
	m_pipeline = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Creation of the RQ pipeline
//
void RayQuery::create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout> rtDescSetLayouts, Scene* scene)
{
	MilliTimer timer;
	LOGI("Create Ray Query Pipeline");

	std::vector<VkPushConstantRange> push_constants;
	push_constants.push_back({ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState) });

	// Create Gbuffer
	m_bufferSize = size.width * size.height;
	m_buffer[0] = m_pAlloc->createBuffer(sizeof(GeomData) * m_bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	NAME_VK(m_buffer[0].buffer);
	m_buffer[1] = m_pAlloc->createBuffer(sizeof(GeomData) * m_bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	NAME_VK(m_buffer[1].buffer);
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
	computePipelineCreateInfo.stage.module = nvvk::createShaderModule(m_device, pathtrace_comp, sizeof(pathtrace_comp));
	computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computePipelineCreateInfo.stage.pName = "main";

	vkCreateComputePipelines(m_device, {}, 1, &computePipelineCreateInfo, nullptr, &m_pipeline);

	m_debug.setObjectName(m_pipeline, "RayQuery");
	vkDestroyShaderModule(m_device, computePipelineCreateInfo.stage.module, nullptr);


	timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
#define GROUP_SIZE 8  // Same group size as in compute shader
void RayQuery::run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets)
{

	// Preparing for the compute shader
	descSets.push_back(m_descSet[(state.frame + 1) % 2]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Sending the push constant information
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &m_state);

	// Dispatching the shader
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

// handle window resize
void RayQuery::update(const VkExtent2D& size) {
	if ((size.width * size.height) > m_bufferSize) {
		m_bufferSize = size.width * size.height;
		m_pAlloc->destroy(m_buffer[0]);
		m_pAlloc->destroy(m_buffer[1]);
		m_buffer[0] = m_pAlloc->createBuffer(sizeof(GeomData) * m_bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		NAME_VK(m_buffer[0].buffer);
		m_buffer[1] = m_pAlloc->createBuffer(sizeof(GeomData) * m_bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		NAME_VK(m_buffer[1].buffer);

		VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
			| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkWriteDescriptorSet, 2> writes;
		writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_dbi[0]);
		writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_dbi[1]);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_dbi[1]);
		writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_dbi[0]);
		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}
}

void RayQuery::createDescriptorSet()
{
	m_bind = nvvk::DescriptorSetBindings{};

	VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	m_bind.addBinding({ RayQBindings::eLastGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size());
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	m_dbi[0] = VkDescriptorBufferInfo{ m_buffer[0].buffer, 0, VK_WHOLE_SIZE };
	m_dbi[1] = VkDescriptorBufferInfo{ m_buffer[1].buffer, 0, VK_WHOLE_SIZE };

	std::array<VkWriteDescriptorSet, 2> writes;
	writes[0] = m_bind.makeWrite(m_descSet[0], RayQBindings::eLastGbuffer, &m_dbi[0]);
	writes[1] = m_bind.makeWrite(m_descSet[0], RayQBindings::eThisGbuffer, &m_dbi[1]);
	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	writes[0] = m_bind.makeWrite(m_descSet[1], RayQBindings::eLastGbuffer, &m_dbi[1]);
	writes[1] = m_bind.makeWrite(m_descSet[1], RayQBindings::eThisGbuffer, &m_dbi[0]);
	vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
