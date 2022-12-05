<div align="center">
    <h1>EIDOLA</h1>
    Real-time GI Path Tracer
</div>

**University of Pennsylvania, [CIS 565: GPU Programming and Architecture, Final Project ](https://cis565-fall-2022.github.io/)**

**Chang Liu ([LinkedIn](https://www.linkedin.com/in/chang-liu-0451a6208/) | [Personal Website](https://hummawhite.github.io/)), Alex Fu ( [LinkedIn](https://www.linkedin.com/in/alex-fu-b47b67238/) |  [Personal Website](https://thecger.com/)) and Yilin Liu ( [LinkedIn](https://www.linkedin.com/in/yilin-liu-9538ba1a5/) |  [Personal Website](https://yilin.games/))**

**Tested on a personal laptop: i7-12700 @ 4.90GHz with 16GB RAM, RTX 3070 Ti Laptop 8GB**

![](./images/bistro_ext.jpg)

## Features

- [ ] Real-time Direct Illumination based on ReSTIR DI
- [ ] Real-time Indirect Illumination based on ReSTIR GI
- [ ] Denoising
- [ ] Displacement Map (still working on)

## Presentations

- [*Pitch*](https://docs.google.com/presentation/d/1NLRpVT09ZlEVntIzB865NTc5noMcgNP8SMXB7Bp0KEk)

- [*Milestone 1*](https://docs.google.com/presentation/d/1NLRpVT09ZlEVntIzB865NTc5noMcgNP8SMXB7Bp0KEk/edit?usp=sharing)

- [*Milestone 2*](https://drive.google.com/file/d/1okQr6V4lZn3Exx-DBy-BUHfuLg3T9ZYt/view)

- [*Milestone 3*](#)

## Introduction

## Pipeline

### Overview

<div align="center">
    <p>(Image)</p>
</div>

### Resampling & Spatiotemporal Approaches

Our implementation largely relies on resampling and spatial & temporal techniques. What is resampling? 

### Direct Illumination

#### G-Buffer

All our ReSTIR and denoising process later require screen space geometry information, so we need to generate G-Buffer. By this time, our G-Buffer is generated along with ray tracing. Here is our G-Buffer layout (we will illustrate this later):

- Depth: float
- Normal: vec3
- Albedo: vec3
- Mat ID: uint
- Material (metallic, roughness, transmission & ior): vec4
- Motion vector: ivec2

To save memory and reduce bandwidth occupancy, our G-Buffer components are compressed with different techniques. After compression, our G-Buffer looks like this:

- Depth: 4 bytes
- Normal: 4 bytes
- Albedo: 3 bytes
- Hashed Mat ID: 1 byte
- Material: 4 bytes
- Motion vector: 4 bytes

Which saves 56 - 20 = 36 bytes per pixel.

#### Direct ReSTIR

...

### Indirect Illumination

#### Indirect ReSTIR

...

#### Reduced Resolution & Block-wise Long Paths

Based on the nature of tracing longer paths and perform more occlusion tests, we observed that tracing indirect lighting is much slower than direct. For example, before we optimized indirect lighting, running the Bistro Exterior scene takes 3.9 ms for direct and 25 ms for indirect.

Wasting time on tracing relatively insignificant component is not effective. Usually the variation of indirect illumination is at lower frequency based on the assumption that most surfaces in a scene are diffuse. With not so sharp variation, we could possibly trace less rays and use interpolation to reconstruct indirect illumination. In our path tracer, we reduce the resolution for indirect lighting to 1/4 of direct lighting and then do "fake" upscaling to produce a full resolution image. The approach we use is blending neighboring pixels, but not as far as bilinear filtering, which we believe could preserve some sharp  details on edges.

Besides, we did as suggested in the ReSTIR GI paper, that to decide whether to trace longer paths with Russian roulette on a block level. We only allow 25% of rays to trace multiple bounces, while the rest 75% trace one bounce.

### Denoising

#### A Lightweight Denoiser based on Edge-Avoiding A-Trous

In both ReSTIR DI and GI we have already included reuse of temporally neighboring samples, which gives us pretty decent temporally stable results. Therefore when it comes to denoising, we don't necessarily need a spatiotemporal denoiser like SVGF, not to say that temporally reused outputs from ReSTIR are correlated and prone to artifacts if denoised temporally.

Just like what we did in project 4, our denoising process is logically divided into three stages: demodulation, filtering and remodulation. We let the output from ReSTIR to be divided by screen-space albedo (trick by setting materials' base color to 1), and do tone mapping to compress radiance values into a range that denoiser can handle well.

The direct and indirect components are filtered separately and merged after filtering. For direct we use a 4-level wavelet filter since it's already smooth. For indirect, we use a 6-level wavelet to reduce flickering.

## Future Improvement

### ...

## Third Party Credit

### Base Code

- [*Vk_Raytrace*](https://github.com/nvpro-samples/vk_raytrace)

### Assets

- [*GLTF Scene: Amazon Lumberyard Bistro*](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)

- [*GLTF Scene: Crytek Sponza*](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Sponza/glTF/Sponza.gltf)

### Referrences

- [Bitterli, Benedikt, et al. "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting." ACM Transactions on Graphics (TOG) 39.4 (2020): 148-1.](https://cs.dartmouth.edu/wjarosz/publications/bitterli20spatiotemporal.html)

-  [Daniel Wright. "Radiance Caching for Real-time Global Illumination." Advances in Real-Time Rendering in Games.SIGGRAPH 2021](https://advances.realtimerendering.com/s2021/index.html)

- [Kajiya Global Illumination Overview](https://github.com/EmbarkStudios/kajiya/blob/main/docs/gi-overview.md)

- [Ouyang, Yaobin, et al. "ReSTIR GI: Path Resampling for Real‐Time Path Tracing." Computer Graphics Forum. Vol. 40. No. 8. 2021.](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)

- [Thonat, Theo, et al. "Tessellation-free displacement mapping for ray tracing." ACM Transactions on Graphics (TOG) 40.6 (2021): 1-16.](https://research.adobe.com/publication/tessellation-free-displacement-mapping-for-ray-tracing/#:~:text=Displacement%20mapping%20is%20a%20powerful,a%20significant%20amount%20of%20memory.)
