---
title: "Paper Review: Handling High FOV Cameras in Vision Networks"
subtitle: How modern neural nets handle fisheye distortion and metric scale
featured_image: /images/blog/computer-vision/beyond-the-pinhole-camera/thumbnail.gif
categories: blog-computer-vision
permalink: /blog/computer-vision/wide-fov-distortion/
---

* TOC
{:toc}

# Introduction

Depth estimation networks struggle with high FOV images since they usually trian with cameras which are limited to ~120 deg FOV (TODO: Please show a comparison of the various FOVs seen in the datasets commonly used for depth estimation). 

Larger FOV cameras also have high distortion which the above training data may not have considered.

## What the degradation actually looks like

That's the argument. Here's the measurement — DAC ran a perspective-trained metric depth model
(Metric3D-v2) on the *same scene* presented five different ways:

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/dac_fisheye_degradation.png" alt="Five panels of the same street scene: raw fisheye at 180 degrees scoring delta-1 0.64, ERP at 180 degrees scoring 0.76, and undistorted perspective crops at 90, 120 and 150 degrees scoring 0.80, 0.66 and 0.45" style="width:100%;border-radius:6px;background:#fff">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    One scene, five representations, one perspective-trained model (Metric3D-v2). Top row: input. Middle: predicted metric depth. Bottom: $\delta_1$ accuracy (higher is better). <em>Figure 2 from Depth Any Camera, arXiv:2501.02464.</em>
  </figcaption>
</figure>

Read the $\delta_1$ row left to right — it's the whole motivation for this post in five numbers:

| Representation | FoV | $\delta_1$ | What it tells you |
|:--|:--:|:--:|:--|
| Raw fisheye | 180° | 0.64 | feed the distorted image straight in and the model struggles |
| **ERP** | 180° | **0.76** | same 180° of content, just re-laid-out — and a big chunk of the loss comes back |
| Undistorted perspective | 90° | 0.80 | the best score, but you threw away most of the field of view to get it |
| Undistorted perspective | 120° | 0.66 | pushing the undistortion wider starts to hurt |
| Undistorted perspective | 150° | 0.45 | now it's far worse than just feeding the raw fisheye |

Three things fall out of that table:

- The "just undistort it" escape hatch works — but only by throwing away FOV. 
- Undistorting *wide* is actively counterproductive. By 150° it scores 0.45, well below the 0.64 you'd get by not undistorting at all
- ERP gets most of it back for free. At the full 180°, ERP scores 0.76 — beating undistorted Perspective at 120° *and* 150°


## Option A: Equi-Rectangular Projection (ERP)

<video width="100%" controls autoplay loop muted playsinline>
  <source src="/images/blog/computer-vision/beyond-the-pinhole-camera/dac-erp-explainer.mp4" type="video/mp4">
</video>

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/dac_pipeline.png" alt="Depth Any Camera pipeline: Image-to-ERP + FoV-Align in training; ERP-to-Image at inference" style="width:100%;border-radius:6px">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    The DAC pipeline (Guo et&nbsp;al., CVPR&nbsp;2025)
  </figcaption>
</figure>


### What ERP buys, and what it costs

- ✅ **Cheap.** A grid-sample sitting in front of whatever depth backbone you already have.
- ✅ **Non-invasive.** The entire loss stack — scale-and-shift-invariant loss, gradient matching,
  temporal consistency — is untouched, because from the network's point of view it's still just
  predicting depth on an image.
- ✅ **Free training data, labels included.** The tilt trick manufactures fisheye supervision out
  of perspective datasets you already have, and the depth maps ride through the same grid.
- ❌ **Needs the camera model at test time** to build the patch — the intrinsics *and* the lens
  model, because the lens is the one step of the chain that isn't pure geometry. For a genuinely
  uncalibrated lens that's a real problem — and it's the opening for the alternative.

## Option B: don't resample, predict the rays (SH ray fields)

UniK3D asks the obvious follow-up: **why resample at all?** Why not let the network predict the
camera's rays directly and skip the intermediate image entirely?

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/unik3d_any_camera.png" alt="Four RGB inputs — equirectangular, Mei fisheye, Fisheye624, and pinhole — all fed into a single UniK3D model, each producing a 3D point cloud" style="width:100%;border-radius:6px;background:#fff">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    <em>Figure 1 from UniK3D, arXiv:2503.16591.</em>
  </figcaption>
</figure>

<video width="100%" controls autoplay loop muted playsinline>
  <source src="/images/blog/computer-vision/beyond-the-pinhole-camera/sh-ray-fields.mp4" type="video/mp4">
</video>

Every camera is fully described by its **pencil of rays**: the map from each pixel to the direction
it looks along. A pinhole's fan out mildly, a fisheye's bend sharply at the edges. Know the ray
field and you know the camera — no intrinsics required.

UniK3D predicts that field as a superposition of **spherical harmonics**. An angular module emits a
small set of coefficients $\mathbf{H}$, and ray directions come back via an inverse SH transform:

$$ \mathbf{C} \;=\; \mathcal{F}_{\mathcal B}^{-1}\{\mathbf H\}
   \;=\; \sum_{l=0}^{L}\sum_{m=-l}^{l} \mathbf H_{lm}\,\mathcal B_{lm}(\theta, \phi). $$

The basis $\mathcal B_{lm}$ is fixed and known — the fifteen tiles in the animation — and only the
coefficients are predicted. Degree 3 with no constant term is 15 harmonics (each carrying a 3-vector
weight in the released code): a remarkably compact way to say "here's how this lens bends light."

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/unik3d_arch.png" alt="UniK3D architecture: Angular Module predicts SH coefficients (rays), Radial Module predicts radial distance" style="width:100%;border-radius:6px">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    UniK3D pencil of rays <em>Figure from arXiv:2503.16591.</em>
  </figcaption>
</figure>

The other choice visible in that figure is the output: **radial distance** along each ray, not
perpendicular depth $Z$, which goes degenerate for a ray pointing sideways. That's what keeps the
representation well-posed past 180° — though it isn't the SH camp's private insight, since DAC
switches to Euclidean distance for the same reason. What stays UniK3D's is that the rays are
*predicted* rather than given.

### What it costs

- ✅ **Calibration-free at test time.** No intrinsics, no lens model, no rectification.
- ✅ **Well-posed beyond 180°** — decisive for genuinely omnidirectional cameras.
- ❌ **Bespoke architecture.** Angular module, radial module conditioned on it, an SH output space,
  and a matching angular loss.
- ❌ **Not a drop-in.** It replaces the head *and* the loss geometry, and emits radial 3D rather
  than the affine-invariant depth map most video-depth loss stacks are built around.

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/unik3d_qualitative.png" alt="Qualitative comparison grid across panoramic, 180-degree Mei fisheye, Fisheye624 and pinhole test sets, showing error maps and reconstructed point clouds for DepthAnything, UniDepth, MASt3R and UniK3D" style="width:100%;border-radius:6px;background:#fff">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    Four test sets (rows: panoramic → 180° fisheye → Fisheye624 → pinhole) against four methods
  </figcaption>
</figure>

## Option C: predict a dense ray map (MapAnything, Depth Anything 3)

- The feed-forward reconstruction
  [MapAnything](https://arxiv.org/abs/2509.13414) and
  [Depth Anything 3](https://arxiv.org/abs/2511.10647) output, per pixel, a **ray direction and a
  depth along that ray**
- MapAnything emits unit ray directions in the local camera frame plus ray depth, a pose and one
  metric scale; DA3 emits a 6-D ray (origin and direction in the world frame) plus depth along it


> **NOTE:** both models are trained and evaluated almost entirely on pinhole data (MapAnything
>  benchmarks on an *undistorted* ETH3D; DA3 evaluates no non-pinhole camera), so their fisheye
>  behaviour is untested rather than solved — MapAnything itself lists fisheye as a matter of
>  "appropriate training".

<figure>
  <img src="/images/blog/computer-vision/beyond-the-pinhole-camera/da3_arch.png" alt="Depth Anything 3 pipeline: images through a single DINO transformer to a dual-DPT head that emits a depth map (N x H x W) and a ray map (N x H/2 x W/2 x 6), combined into points; an optional camera token enters at the input and a small decoder recovers camera parameters" style="width:100%;border-radius:6px;background:#fff">
  <figcaption style="text-align:center;color:#8b8fa3;font-size:13px">
    Depth Anything 3 uses camera intrinsics if known, but only as an optional token. <em>Figure 2 from Depth Anything 3, arXiv:2511.10647.</em>
  </figcaption>
</figure>

This representation of ray maps gives you UniK3D's calibration-free deployment without the SH machinery.

## Side by side

| | **A · ERP / gnomonic** | **B · SH ray field** | **C · dense ray map** |
|:--|:--|:--|:--|
| What it is | resampling front-end | learned camera, 15 weights | learned camera, one ray per pixel |
| Touches the backbone? | no — grid-sample in front | yes — replaces head *and* losses | yes — but reconstruction models already have the head |
| Needs intrinsics at test time? | **yes** | no | no |
| Evaluated on non-pinhole cameras? | yes | yes | not yet |
| Valid past 180°? | no | **yes** | yes (depth along the ray) |
| Camera prior | exact, supplied | smooth, low-order | none |
| Representative work | Depth Any Camera | UniK3D, CAM3R | MapAnything, Depth Anything 3 |

> **In one line.** ERP is a cheap front-end that preserves your whole pipeline; SH rays are a more
> powerful but more invasive rebuild; a dense ray map is that rebuild already paid for by the
> reconstruction models.

---

# Part 4: Which one should you reach for?

```mermaid
flowchart TD
    Q1["<b>Field of view &gt; 180°?</b><br/><i>omnidirectional / panoramic</i>"]
    Q2["<b>Can the lens be calibrated?</b><br/><i>even once, offline</i>"]
    Q3["<b>Monocular depth, or<br/>feed-forward reconstruction?</b>"]
    ERP["<b>ERP front-end</b><br/>Depth Any Camera<br/><i>grid-sample, no retrain</i>"]
    PM["<b>dense ray map + ray depth</b><br/>MapAnything · DA3<br/><i>option C — calibration-free</i>"]
    SH["<b>SH ray field</b><br/>UniK3D · CAM3R<br/><i>calibration-free,<br/>well-posed past 180°</i>"]

    Q1 -- "no" --> Q2
    Q1 -- "yes" --> SH
    Q2 -- "yes, once" --> Q3
    Q2 -- "never" --> SH
    Q3 -- "depth" --> ERP
    Q3 -- "reconstruction" --> PM

    classDef q fill:#1a1d27,stroke:#6c9eff,color:#e0e0e6;
    classDef rec fill:#17301f,stroke:#2e8b57,color:#e0e0e6;
    classDef sh fill:#2a2340,stroke:#8b5cf6,color:#e0e0e6;
    class Q1,Q2,Q3 q;
    class ERP,PM rec;
    class SH sh;
```

<div style="text-align:center;color:#8b8fa3;font-size:13px;margin:-0.5em 0 2em">Three questions decide it. Below 180° and calibratable, the depth path takes ERP (A) and the feed-forward reconstruction path keeps its dense ray-map head (C); SH rays (B) are for the &gt;180° or never-calibrated branches.</div>

Take the case that motivates most of this work: train on ~80–90° perspective data, deploy on a ~140°
fisheye. **ERP is enough, and probably the right call.**

1. **140° is comfortably below 180°.** SH's headline advantage is staying well-posed *past* 180°;
   below it, ERP/gnomonic is fully well-posed across the whole cone. The advantage isn't in play.
2. **ERP is non-invasive; SH rays are a rebuild.** A grid-sample in front of the backbone leaves
   architecture and losses untouched — a much better fit for a low-data adaptation regime.
3. **Calibration is a one-time cost.** ERP's one real weakness is needing the camera model, and you
   can calibrate the lens once (Kannala–Brandt, or the vendor model) and plug it in **with no
   retraining**.


# References

- Guo et al., *Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera*, CVPR 2025 — [arXiv:2501.02464](https://arxiv.org/abs/2501.02464)
- Piccinelli et al., *UniK3D: Universal Camera Monocular 3D Estimation*, CVPR 2025 — [arXiv:2503.16591](https://arxiv.org/abs/2503.16591)
- Yin et al., *Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image*, ICCV 2023 — [arXiv:2307.10984](https://arxiv.org/abs/2307.10984)
- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 — [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
- *Calibration Tokens: Adapting Foundation Depth Models for Fisheye Cameras* — [arXiv:2508.04928](https://arxiv.org/abs/2508.04928)
- Keetha et al., *MapAnything: Universal Feed-Forward Metric 3D Reconstruction*, 2025 — [arXiv:2509.13414](https://arxiv.org/abs/2509.13414)
- Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views*, 2025 — [arXiv:2511.10647](https://arxiv.org/abs/2511.10647)

All paper figures above are reproduced from the linked arXiv preprints and are credited in their
captions.
