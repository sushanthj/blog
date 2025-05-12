---
layout: knowledge
title: Camera Models
parent: Computer Vision
nav_order: 1
permalink: /knowledge-base/computer-vision/camera-model/
toc: true
---

* TOC
{:toc}

# Introduction

## Basics

![](../../../images/knowledge_base/computer_vision/camera_models/13.png)

![](../../../images/knowledge_base/computer_vision/camera_models/2.png)

![](../../../images/knowledge_base/computer_vision/camera_models/3.png)

In the above image, the division by Z happens implicitly due to homogenous coordinate notation

## Account for other issues in image frame

**We will introduce 3 coordinate systems below:**
1. Camera Coordinate Frame
2. Image Coordinate Frame (where homogenous notation is used as there is no z-axis information)
3. World Coordinate Information

Sometimes the camera coordinate frame and the image coordinate frame is misaligned as shown below:

![](../../../images/knowledge_base/computer_vision/camera_models/4.png)

![](../../../images/knowledge_base/computer_vision/camera_models/5.png)

## Intrinsic and Extrinsic Decomposition

![](../../../images/knowledge_base/computer_vision/camera_models/6.png)

![](../../../images/knowledge_base/computer_vision/camera_models/7.png)

![](../../../images/knowledge_base/computer_vision/camera_models/8.png)

### Lesson Learnt:

If we follow the how a 3D point gets left multiplied by extrinsic and then by intrinsic the
coordinate frame intuition we derive is:

(3D Point -> Extrinsic -> Intrinsic)  =  (World Frame -> Camera Frame -> Image Frame)

![](../../../images/knowledge_base/computer_vision/camera_models/9.png)

t = Translation (last column of extrinsic matrix)
R = Rotation (first 3x3 part of extrinsic matrix)

## Final Version of Camera Model (I prefer this)

![](../../../images/knowledge_base/computer_vision/camera_models/12.png)

![](../../../images/knowledge_base/computer_vision/camera_models/11.jpg)