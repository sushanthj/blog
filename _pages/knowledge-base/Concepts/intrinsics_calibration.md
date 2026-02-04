---
layout: knowledge
title: Camera Intrinsics Calibration
parent: Concepts
nav_order: 3
permalink: /knowledge-base/computer-vision/camera-intrinsics-calibration/
toc: true
---

* TOC
{:toc}

# Intrinsics Calibration

In this we estimate the 3x3 Intrinsic matrix (K) and the distortion coefficients. The math is as follows:

## Projection Pipeline

The complete projection from a 3D world point to a 2D pixel coordinate follows these steps:

![Projection](/images/knowledge_base/concepts/computer_vision/camera_calibration/projection.png)

![Forward Projection Flowchart](/images/knowledge_base/concepts/computer_vision/camera_calibration/forward_projection_flowchart.svg)

```
Pixel Space ← Distorted Space ← Normalized Image Space ←  Camera Space   ← World Space
   (x, y)        (x_d, y_d)           (x_n, y_n)          (X_c,Y_c,Z_c)   (X,Y,Z)
      K           distortion              ÷ Z_c             [R | t]
```

---

### Step 1: World Space → Camera Space

Transform the 3D world point **(X, Y, Z)** into the camera coordinate frame using the **extrinsic matrix [R | t]**

```
⎡ X_c ⎤       ⎡ r₁₁  r₁₂  r₁₃ ⎤   ⎡ X ⎤     ⎡ t_x ⎤
⎢ Y_c ⎥   =   ⎢ r₂₁  r₂₂  r₂₃ ⎥   ⎢ Y ⎥  +  ⎢ t_y ⎥
⎣ Z_c ⎦       ⎣ r₃₁  r₃₂  r₃₃ ⎦   ⎣ Z ⎦     ⎣ t_z ⎦
```

Or compactly: **X_c = R @ X + t**

---

### Step 2: Camera Space → Normalized Image Space

Project onto the normalized image plane (Z = 1) using perspective division:

```
x_n = X_c / Z_c
y_n = Y_c / Z_c
```

This gives us the ideal pinhole projection, ignoring lens effects.

---

### Step 3: Normalized Space → Distorted Space

Apply lens distortion to the normalized coordinates. Different camera types use different distortion models:

#### 3A: Pinhole Camera Distortion Model

For standard pinhole cameras (rectilinear lenses):

- **Radial distortion**: k₁, k₂, k₃ (barrel/pincushion effects)
- **Tangential distortion**: p₁, p₂ (lens misalignment)

```
x_d = x_n(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁x_n·y_n + p₂(r² + 2x_n²)

y_d = y_n(1 + k₁r² + k₂r⁴ + k₃r⁶) + p₁(r² + 2y_n²) + 2p₂x_n·y_n
```

Where **r² = x_n² + y_n²** (squared distance from optical axis).

#### 3B: Fisheye Camera Distortion Model

For fisheye cameras (wide-angle lenses):

- **Radial distortion**: k₁, k₂, k₃, k₄ (stronger radial effects)
- No tangential distortion (fisheye lenses typically don't have this)

```
r = √(x_n² + y_n²)
θ = atan(r)  # Angle from optical axis
θ_d = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)  # Distorted angle
r_d = tan(θ_d)  # Distorted radius

x_d = (r_d / r) · x_n
y_d = (r_d / r) · y_n
```

Where **r = √(x_n² + y_n²)** (distance from optical axis).

---

### Step 4: Distorted Space → Pixel Space

Apply the **intrinsic matrix K** to convert to final pixel coordinates:

```
⎡ x ⎤       ⎡ s_x  s_θ   o_x ⎤   ⎡ x_d ⎤
⎢ y ⎥   =   ⎢ 0    s_y   o_y ⎥   ⎢ y_d ⎥
⎣ 1 ⎦       ⎣ 0     0    1   ⎦   ⎣  1  ⎦
```

Or compactly: **x = K @ x_d**

Where:
- **s_x, s_y** — focal lengths in pixels
- **s_θ** — skew coefficient (typically 0)
- **(o_x, o_y)** — principal point (optical center)

---

## Compact Forms

### Ideal Pinhole (No Distortion)

For an ideal camera without distortion, the full projection simplifies to:

```
    ⎡ x ⎤       ⎡ s_x  s_θ   o_x ⎤   ⎡ r₁₁  r₁₂  r₁₃  t_x ⎤   ⎡ X ⎤
λ   ⎢ y ⎥   =   ⎢ 0    s_y   o_y ⎥   ⎢ r₂₁  r₂₂  r₂₃  t_y ⎥   ⎢ Y ⎥
    ⎣ 1 ⎦       ⎣ 0     0    1   ⎦   ⎣ r₃₁  r₃₂  r₃₃  t_z ⎦   ⎢ Z ⎥
                                                              ⎣ 1 ⎦
```

### Real Camera (With Distortion)

For a real camera with lens distortion, the projection becomes:

```
x = K @ distort(normalize([R | t] @ X))
```

## Undistortion

**Requirement**: Given distorted coordinates (x_d, y_d), find the original normalized coordinates (x_n, y_n).

**Problem**: The distortion equations are nonlinear and cannot be inverted analytically. Use iterative fixed-point method:

### Undistort Pinhole Camera

Remember:
```
x_d = x_n(1 + k₁r² + k₂r⁴ + k₃r⁶) + 2p₁x_n·y_n + p₂(r² + 2x_n²)

y_d = y_n(1 + k₁r² + k₂r⁴ + k₃r⁶) + p₁(r² + 2y_n²) + 2p₂x_n·y_n
```

Undistort:
```
x_n, y_n = x_d, y_d  # Initial guess
tolerance = 1e-6

while True:
    x_n_prev, y_n_prev = x_n, y_n
    
    r² = x_n² + y_n²
    radial = 1 + k₁r² + k₂r⁴ + k₃r⁶
    x_n = (x_d - 2p₁x_n·y_n - p₂(r² + 2x_n²)) / radial
    y_n = (y_d - p₁(r² + 2y_n²) - 2p₂x_n·y_n) / radial
    
    # Stop when change is negligible
    if |x_n - x_n_prev| < tolerance and |y_n - y_n_prev| < tolerance:
        break
```

**OpenCV function**: `cv2.undistortPoints()`

### Undistort Fisheye Camera

Remember:
```
r = √(x_n² + y_n²)
θ = atan(r)
θ_d = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
r_d = tan(θ_d)

x_d = (r_d / r) · x_n
y_d = (r_d / r) · y_n
```

Undistort:
```
r_d = √(x_d² + y_d²)
θ_d = atan(r_d)
θ = θ_d  # Initial guess
tolerance = 1e-6

while True:
    θ_prev = θ
    θ = θ_d / (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
    
    # Stop when change is negligible
    if |θ - θ_prev| < tolerance:
        break

r = tan(θ)
x_n = (r / r_d) · x_d
y_n = (r / r_d) · y_d
```

**OpenCV function**: `cv2.fisheye.undistortPoints()`

