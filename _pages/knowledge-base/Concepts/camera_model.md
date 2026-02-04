---
layout: knowledge
title: Camera Models
parent: Concepts
nav_order: 1
permalink: /knowledge-base/computer-vision/camera-model/
toc: true
---

* TOC
{:toc}

# Introduction

## Basics

Even without going into details, the most barebones camera model can be represented as:

![](/images/knowledge_base/concepts/computer_vision/camera_models/13.png)


## Intrinsics and Extrinsics

The above 3x4 matrix actually represents a combination of intrinsic and extrinsic parameters.

![](/images/knowledge_base/concepts/computer_vision/camera_models/12.png)

Even further this can be decomposed into:

![](/images/knowledge_base/concepts/computer_vision/camera_models/11.jpg)
## Three Coordinate Systems

The camera projection model involves three coordinate systems:
1. Camera Coordinate Frame
2. Image Coordinate Frame (where homogenous notation is used as there is no z-axis information)
3. World Coordinate Information

Sometimes the camera coordinate frame and the image coordinate frame is misaligned as shown below:

![](/images/knowledge_base/concepts/computer_vision/camera_models/4.png)

![](/images/knowledge_base/concepts/computer_vision/camera_models/5.png)


