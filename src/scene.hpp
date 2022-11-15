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


 //--------------------------------------------------------------------------------------------------
 // - Loading and storing the glTF scene
 // - Creates the buffers and descriptor set for the scene


#include <string>

#include "nvh/gltfscene.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "queue.hpp"

class Scene
{
public:
	enum EBuffer
	{
		eCameraMat,
		eMaterial,
		eInstData,
		ePuncLights,
		eTrigLights,
		// eTrigLightTransforms,
		eLightBufInfo,
		//eGbuffer
	};


	enum EBuffers
	{
		eVertex,
		eIndex,
	};

public:
	void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const nvvk::Queue& queue, nvvk::ResourceAllocator* allocator);
	bool load(const std::string& filename);

	void createInstanceDataBuffer(VkCommandBuffer cmdBuf, nvh::GltfScene& gltf);
	void createVertexBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf);
	void setCameraFromScene(const std::string& filename, const nvh::GltfScene& gltf);
	bool loadGltfScene(const std::string& filename, tinygltf::Model& tmodel);
	void createPuncLightBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf);
	void createTrigLightBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf, const tinygltf::Model& gltfModel);
	void createMaterialBuffer(VkCommandBuffer cmdBuf, const nvh::GltfScene& gltf);
	void destroy();
	void updateCamera(const VkCommandBuffer& cmdBuf, VkExtent2D size);


	VkDescriptorSetLayout            getDescLayout() { return m_descSetLayout; }
	VkDescriptorSet                  getDescSet() { return m_descSet; }
	nvh::GltfScene& getScene() { return m_gltf; }
	nvh::GltfStats& getStat() { return m_stats; }
	const std::vector<nvvk::Buffer>& getBuffers(EBuffers b) { return m_buffers[b]; }
	const std::string& getSceneName() const { return m_sceneName; }
	SceneCamera& getCamera() { return m_camera; }
private:
	void createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel);
	void createDescriptorSet(const nvh::GltfScene& gltf);

	nvh::GltfScene m_gltf;
	nvh::GltfStats m_stats;

	std::string m_sceneName;
	SceneCamera m_camera{};

	// Setup
	nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
	nvvk::DebugUtil          m_debug;   // Utility to name objects
	VkDevice                 m_device;
	nvvk::Queue              m_queue;

	// Resources
	std::array<nvvk::Buffer, 6>                            m_buffer;           // For single buffer
	std::array<std::vector<nvvk::Buffer>, 2>               m_buffers;          // For array of buffers (vertex/index)
	std::vector<nvvk::Texture>                             m_textures;         // vector of all textures of the scene
	std::vector<std::pair<nvvk::Image, VkImageCreateInfo>> m_images;           // vector of all images of the scene
	std::vector<size_t>                                    m_defaultTextures;  // for cleanup


	VkDescriptorPool      m_descPool{ VK_NULL_HANDLE };
	VkDescriptorSetLayout m_descSetLayout{ VK_NULL_HANDLE };
	VkDescriptorSet       m_descSet{ VK_NULL_HANDLE };

	// Direct light importance sampling
	LightBufInfo m_lightBufInfo;
	float m_puncLightWeight{ 0.f }, m_trigLightWeight{ 0.f };
	float createPuncLightImptSampAccel(std::vector<PuncLight>& puncLights, const nvh::GltfScene& gltf);
	float createTrigLightImptSampAccel(std::vector<TrigLight>& trigLights, const nvh::GltfScene& gltf, const tinygltf::Model& gltfModel);
	float computeTrigIntensity(const TrigLight& trig, const nvh::GltfMaterial& mtl, const tinygltf::Model& gltfModel);
};
