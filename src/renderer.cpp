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

#include "autogen/direct_stage.comp.h"
#include "autogen/direct_gen.comp.h"
#include "autogen/direct_reuse.comp.h"
#include "autogen/indirect_stage.comp.h"
#include "autogen/denoise_direct.comp.h"
#include "autogen/denoise_indirect.comp.h"
#include "autogen/compose.comp.h"

VkPipeline createComputePipeline(VkDevice device, VkComputePipelineCreateInfo createInfo, const uint32_t* shader, size_t bytes) {
	VkPipeline pipeline;
	auto shaderModule = nvvk::createShaderModule(device, shader, bytes);
	createInfo.stage.module = shaderModule;
	vkCreateComputePipelines(device, {}, 1, &createInfo, nullptr, &pipeline);
	vkDestroyShaderModule(device, shaderModule, nullptr);
	return pipeline;
}

void Renderer::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator, uint32_t imageCount)
{
	m_device = device;
	m_imageCount = imageCount;
	m_pAlloc = allocator;
	m_queueIndex = familyIndex;
	m_debug.setup(device);
}

void Renderer::destroy()
{
	for (int i = 0; i < 2; i++) {
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_directReservoir[i]);
		m_pAlloc->destroy(m_indirectReservoir[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i]);
	}
	m_pAlloc->destroy(m_directTempResv);
	m_pAlloc->destroy(m_indirectTempResv);
	m_pAlloc->destroy(m_motionVector);

	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	auto destroyPipeline = [&](VkPipeline& pipeline) {
		vkDestroyPipeline(m_device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
	};

	destroyPipeline(m_directPipeline);
	destroyPipeline(m_directGenPipeline);
	destroyPipeline(m_directReusePipeline);
	destroyPipeline(m_indirectPipeline);
	destroyPipeline(m_denoiseDirectPipeline);
	destroyPipeline(m_denoiseIndirectPipeline);
	destroyPipeline(m_composePipeline);

	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
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

	VkComputePipelineCreateInfo createInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	createInfo.layout = m_pipelineLayout;
	createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	createInfo.stage.pName = "main";

	m_directPipeline = createComputePipeline(m_device, createInfo, direct_stage_comp, sizeof(direct_stage_comp));
	m_debug.setObjectName(m_directPipeline, "Renderer-Direct");

	m_directGenPipeline = createComputePipeline(m_device, createInfo, direct_gen_comp, sizeof(direct_gen_comp));
	m_debug.setObjectName(m_directGenPipeline, "Render-Direct-Gen");

	m_directReusePipeline = createComputePipeline(m_device, createInfo, direct_reuse_comp, sizeof(direct_reuse_comp));
	m_debug.setObjectName(m_directReusePipeline, "Render-Direct-Reuse");

	m_indirectPipeline = createComputePipeline(m_device, createInfo, indirect_stage_comp, sizeof(indirect_stage_comp));
	m_debug.setObjectName(m_indirectPipeline, "Renderer-Indirect");

	m_denoiseDirectPipeline = createComputePipeline(m_device, createInfo, denoise_direct_comp, sizeof(denoise_direct_comp));
	m_debug.setObjectName(m_denoiseDirectPipeline, "Denoise Direct");

	m_denoiseIndirectPipeline = createComputePipeline(m_device, createInfo, denoise_indirect_comp, sizeof(denoise_indirect_comp));
	m_debug.setObjectName(m_denoiseIndirectPipeline, "Denoise Indirect");

	m_composePipeline = createComputePipeline(m_device, createInfo, compose_comp, sizeof(compose_comp));
	m_debug.setObjectName(m_composePipeline, "Compose");

	timer.print();
}


//--------------------------------------------------------------------------------------------------
// Executing the Ray Query compute shader
//
void Renderer::run(const VkCommandBuffer& cmdBuf, const RtxState& state, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet> descSets, int frames)
{
	RtxState cState = state;
	// Preparing for the compute shader
	descSets.push_back(m_descSet[(frames + 1) % 2]);
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Sending the push constant information
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &state);

	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directPipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], RayTraceBlockSizeX), CEIL_DIV(state.size[1], RayTraceBlockSizeY), 1);

	/*
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directGenPipeline);
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_directReusePipeline);
	vkCmdDispatch(cmdBuf, (state.size[0] + (GROUP_SIZE - 1)) / GROUP_SIZE, (state.size[1] + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	*/

	ivec2 indSize = state.size / 2;
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(indSize[0], RayTraceBlockSizeX), CEIL_DIV(indSize[1], RayTraceBlockSizeY), 1);

	if (state.denoise != 0) {
		for (int i = 0; i < 4; i++) {
			cState.denoiseLevel = i;
			vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &cState);
			vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiseDirectPipeline);
			vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], DenoiseBlockSizeX), CEIL_DIV(state.size[1], DenoiseBlockSizeY), 1);
		}
	}
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiseIndirectPipeline);
#if !DENOISER_INDIRECT_BILATERAL
	const int IndirectDenoiseNum = 5;
	ivec2 indDenoiseSize = indSize;

	for (int i = 0; i < IndirectDenoiseNum; i++) {
		cState.denoiseLevel = i;
		vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &cState);
		vkCmdDispatch(cmdBuf, CEIL_DIV(indDenoiseSize[0], DenoiseBlockSizeX), CEIL_DIV(indDenoiseSize[1], DenoiseBlockSizeY), 1);
	}
#else
	ivec2 indDenoiseSize = indSize;
	vkCmdDispatch(cmdBuf, CEIL_DIV(indDenoiseSize[0], DenoiseBlockSizeX), CEIL_DIV(indDenoiseSize[1], DenoiseBlockSizeY), 1);
#endif
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_composePipeline);
	vkCmdDispatch(cmdBuf, CEIL_DIV(state.size[0], ComposeBlockSizeX), CEIL_DIV(state.size[1], ComposeBlockSizeY), 1);
}

// handle window resize
void Renderer::update(const VkExtent2D& size) {
	m_size = size;
	for (int i = 0; i < 2; i++) {
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_directReservoir[i]);
		m_pAlloc->destroy(m_indirectReservoir[i]);
		m_pAlloc->destroy(m_denoiseTempBuf[i]);
	}
	m_pAlloc->destroy(m_directTempResv);
	m_pAlloc->destroy(m_indirectTempResv);
	m_pAlloc->destroy(m_motionVector);

	createImage();
	createBuffer();
	updateDescriptorSet();
}

void Renderer::createBuffer()
{
	VkDeviceSize directSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectSize = (m_size.width / 2) * (m_size.height / 2) * sizeof(IndirectReservoir);
	for (int i = 0; i < 2; i++) {
		m_directReservoir[i] = m_pAlloc->createBuffer(directSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_indirectReservoir[i] = m_pAlloc->createBuffer(indirectSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
	m_directTempResv = m_pAlloc->createBuffer(directSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_indirectTempResv = m_pAlloc->createBuffer(indirectSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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

		auto motionVecCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_motionVectorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
		nvvk::Image motionVecImg = m_pAlloc->createImage(motionVecCreateInfo);
		NAME_VK(motionVecImg.image);

		VkImageViewCreateInfo mvivInfo = nvvk::makeImageViewCreateInfo(motionVecImg.image, motionVecCreateInfo);
		m_motionVector = m_pAlloc->createTexture(motionVecImg, mvivInfo);
		m_motionVector.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		auto denoiseBufInfo = nvvk::makeImage2DCreateInfo(m_size, m_denoiseTempFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, false);
		nvvk::Image denoiseImage1 = m_pAlloc->createImage(denoiseBufInfo);
		nvvk::Image denoiseImage2 = m_pAlloc->createImage(denoiseBufInfo);
		VkImageViewCreateInfo divInfo1 = nvvk::makeImageViewCreateInfo(denoiseImage1.image, denoiseBufInfo);
		VkImageViewCreateInfo divInfo2 = nvvk::makeImageViewCreateInfo(denoiseImage2.image, denoiseBufInfo);

		m_denoiseTempBuf[0] = m_pAlloc->createTexture(denoiseImage1, divInfo1);
		m_denoiseTempBuf[1] = m_pAlloc->createTexture(denoiseImage2, divInfo2);
		m_denoiseTempBuf[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_denoiseTempBuf[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_motionVector.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_denoiseTempBuf[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		
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

	m_bind.addBinding({ RayQBindings::eLastDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eTempDirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ RayQBindings::eLastIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eThisIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ RayQBindings::eTempIndirectResv, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ RayQBindings::eMotionVector, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_bind.addBinding({ RayQBindings::eDenoiseTempA, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ RayQBindings::eDenoiseTempB, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size());
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	updateDescriptorSet();
}

void Renderer::updateDescriptorSet() {
	std::array<VkWriteDescriptorSet, 11> writes;
	VkDeviceSize directResvSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectResvSize = (m_size.width / 2) * (m_size.height / 2) * sizeof(IndirectReservoir);

	for (int i = 0; i < 2; i++) {
		VkDescriptorBufferInfo lastDirectResvBufInfo = { m_directReservoir[i].buffer, 0, directResvSize };
		VkDescriptorBufferInfo thisDirectResvBufInfo = { m_directReservoir[!i].buffer, 0, directResvSize };
		VkDescriptorBufferInfo tempDirectResvBufInfo = { m_directTempResv.buffer, 0, directResvSize };

		VkDescriptorBufferInfo lastIndirectResvBufInfo = { m_indirectReservoir[i].buffer, 0, indirectResvSize };
		VkDescriptorBufferInfo thisIndirectResvBufInfo = { m_indirectReservoir[!i].buffer, 0, indirectResvSize };
		VkDescriptorBufferInfo tempIndirectResvBufInfo = { m_indirectTempResv.buffer, 0, indirectResvSize };

		writes[0] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastGbuffer, &m_gbuffer[i].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisGbuffer, &m_gbuffer[!i].descriptor);

		writes[2] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastDirectResv, &lastDirectResvBufInfo);
		writes[3] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisDirectResv, &thisDirectResvBufInfo);
		writes[4] = m_bind.makeWrite(m_descSet[i], RayQBindings::eTempDirectResv, &tempDirectResvBufInfo);

		writes[5] = m_bind.makeWrite(m_descSet[i], RayQBindings::eLastIndirectResv, &lastIndirectResvBufInfo);
		writes[6] = m_bind.makeWrite(m_descSet[i], RayQBindings::eThisIndirectResv, &thisIndirectResvBufInfo);
		writes[7] = m_bind.makeWrite(m_descSet[i], RayQBindings::eTempIndirectResv, &tempIndirectResvBufInfo);

		writes[8] = m_bind.makeWrite(m_descSet[i], RayQBindings::eMotionVector, &m_motionVector.descriptor);

		writes[9] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseTempA, &m_denoiseTempBuf[0].descriptor);
		writes[10] = m_bind.makeWrite(m_descSet[i], RayQBindings::eDenoiseTempB, &m_denoiseTempBuf[1].descriptor);

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}
}
