---
layout: projects-post
title: Adaptive Street View
subtitle: LiDAR NeRFs + Diffusion
featured_image: /images/projects/adaptive_street_view/cover.gif
categories: projects
---

We explore the application of Neural Radiance Fields to generate novel street views, similar to the likes of Google Street View but without the need for special data acquisition and curation by leveraging the vast and open-source self-driving datasets like Argoverse2. To address the challenges posed by sparse camera views and biased straight line trajectories of the viewpoints in these datasets we use LiDAR maps as depths priors to help the NeRF model converge faster while also learning point embeddings for each location of the LiDAR point cloud. Furthermore, we propose diffusion guided optimization to modify these embeddings allowing us to render custom textures in the scene from user-defined prompts while maintaining geometric consistency across views. We test our approach on the Argoverse2 dataset and provide a qualitative as well as a quantitative comparison of the rendered scene against the ground truth images

[🐙 Github Link](https://github.com/sushanthj/adaptive-street-view)

![](/images/projects/adaptive_street_view/architecture.png)

| **Left View**   |  **Front View**                         |**Right View**                         |
|:------------------------|:------------------------------------|:------------------------------------|
| <video width="100%" autoplay loop muted playsinline><source src="/images/projects/adaptive_street_view/combined_seq_left.mp4" type="video/mp4"></video> | <video width="100%" autoplay loop muted playsinline><source src="/images/projects/adaptive_street_view/combined_seq_front.mp4" type="video/mp4"></video> | <video width="100%" autoplay loop muted playsinline><source src="/images/projects/adaptive_street_view/combined_seq_right.mp4" type="video/mp4"></video> |

## Results

### ControlNet Comparisons

![](/images/projects/adaptive_street_view/full_comp.png)