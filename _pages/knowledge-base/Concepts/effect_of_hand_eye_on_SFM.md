---
layout: knowledge
title: SFM before and after Hand-Eye Calibration
parent: Concepts
nav_order: 4
permalink: /knowledge-base/computer-vision/sfm-before-after-hand-eye-calibration/
toc: true
---

NOTE: Please read the [Hand-Eye Calibration](./hand-eye-calibration/) page first for context.

# Effect of Extrinsic Calibration Errors on SFM/Triangulation

This analyzes how errors in `T_EndEffector_to_Camera` affect downstream triangulation and 3D reconstruction accuracy.


Camera pose is computed as an extension to forward kinematics:
```
T_robot_to_camera[i] = T_robot_to_EndEffector[i] @ T_EndEffector_to_Camera
```

If we approximate `T_EndEffector_to_Camera ≈ I` (identity), any actual offset or rotation introduces systematic error into all camera poses.

---

## Refreshers: Camera Geometry

![Coorindate Systems](/images/knowledge_base/concepts/computer_vision/camera_calibration/camera_coordinate_systems.png)

![Projection](/images/knowledge_base/concepts/computer_vision/camera_calibration/projection.png)

### Forward Projection (3D → 2D)

![Forward Projection Flowchart](/images/knowledge_base/concepts/computer_vision/camera_calibration/forward_projection_flowchart.svg)

### Inverse Projection (2D → ray in 3D)

![Inverse Projection Flowchart](/images/knowledge_base/concepts/computer_vision/camera_calibration/inverse_projection_flowchart.svg)
---

## Experiment Setup

```
Working distance:  d = 50 mm
Baseline:          b = 30 mm
```

---

### Triangulation Background

Given a 3D point P observed by two cameras, triangulation recovers P from:
1. Camera centers C₁, C₂ (in world frame)
2. Ray directions r₁, r₂ (in world frame, derived from pixel coordinates)

**NOTE: pixel coordinates give us the vector from the camera center to the point in the camera frame. Hence, that itself is the ray direction in the camera frame.**

**Pixel -> ray direction**
```
Step 1: Convert pixel to normalized coordinates

⎡ x_n ⎤         ⎡ x ⎤
⎢ y_n ⎥ = K⁻¹ @ ⎢ y ⎥    (then undistort if needed)
⎣  1  ⎦         ⎣ 1 ⎦

Step 2: Ray direction in camera frame

r_cam = [x_n, y_n, 1]  (this vector points from camera center toward the 3D point at depth = 1)

Step 3: Transform to world frame

r_world = R @ r_cam    (where R is camera rotation in world frame)
```

**Triangulation Methods:**

### Method 1: Ray Intersection

Each camera observes the 3D point P and gives us a ray. A ray is defined by:
- **Origin**: Camera center C (where the ray starts)
- **Direction**: Unit vector r (direction the ray points)

Any point along the ray can be written as:
```
L(s) = origin + s × direction (where s ≥ 0)
L(s) = C + s × r    
```

- When s = 0: L(0) = C (at the camera center)
- When s = 1: L(1) = C + r (one unit along the ray)
- When s = 50: L(50) = C + 50×r (50 units along the ray, i.e., at depth 50)

For two cameras, we have two rays:
```
L₁(s) = C₁ + s × r₁    (ray from camera 1)
L₂(t) = C₂ + t × r₂    (ray from camera 2)
```

Ideally, both rays intersect at the 3D point P. In practice (due to noise), they are skew lines that don't exactly intersect. We find s and t that minimize the distance between the rays:
```
minimize ||L₁(s) - L₂(t)||²
```

The reconstructed point is the midpoint of the closest approach:
```
P_reconstructed = (L₁(s*) + L₂(t*)) / 2
```

### Method 2: DLT (Direct Linear Transform)

Instead of ray intersection, we can use the projection equation directly.

**The problem:** We have `x = P @ X` (pixel = camera_matrix @ 3D_point), but we can't directly invert because of scale ambiguity: `x = λ @ P @ X` for unknown λ.

**The solution:** Eliminate λ using cross product. If `x = λ @ P @ X`, then `x × (P @ X) = 0`.

**Why?** Both `x` (pixel in homogeneous coords) and `P @ X` (projected 3D point) represent vectors from the camera center through the same pixel on the image plane. They point in the **same direction** — only differ by scale λ. The cross product of parallel vectors is zero.

![DLT Cross Product Intuition](/images/knowledge_base/concepts/computer_vision/camera_calibration/dlt_cross_product_intuition.svg)

For a pixel (x, y) and camera matrix P with rows p₁, p₂, p₃:
```
⎡ y @ p₃ - p₂ ⎤       ⎡ 0 ⎤
⎢ p₁ - x @ p₃ ⎥ @ X = ⎢ 0 ⎥
⎣     ...     ⎦       ⎣ 0 ⎦
```

Each point correspondence gives 2 equations. Stack equations from both cameras:
```
⎡ y₁ @ p₃¹ - p₂¹  ⎤       ⎡ 0 ⎤
⎢ p₁¹ - x₁ @ p₃¹  ⎥       ⎢ 0 ⎥
⎢ y₂ @ p₃² - p₂²  ⎥ @ X = ⎢ 0 ⎥
⎣ p₁² - x₂ @ p₃²  ⎦       ⎣ 0 ⎦

       A (4×4)      X (4×1)
```

This is a homogeneous system `A @ X = 0`. Solution: X is the null space of A, found via SVD as the last column of V. 

NOTE: The last column of V (corresponds to the 3rd singular value ~ 0) is the direction in which all space gets collapsed to a point -> i.e. the null space. 

```python
A = np.array([
    y1 * C1[2,:] - C1[1,:],
    C1[0,:] - x1 * C1[2,:],
    y2 * C2[2,:] - C2[1,:],
    C2[0,:] - x2 * C2[2,:]
])
U, S, V_T = np.linalg.svd(A)
X = V_T.T[:, -1]             # last column of V
X = X / X[3]                 # convert from homogeneous to 3D
P_reconstructed = X[:3]
```

---

### Case 1: Translation Error Only

|                       | Assumed       | True          |
|-----------------------|---------------|---------------|
| Translation component | [0, 0, 0]ᵀ    | [2, 0, 0]ᵀ    |

**Assumed T_EndEffector_to_Camera:**
```
⎡ 1  0  0  0 ⎤
⎢ 0  1  0  0 ⎥
⎢ 0  0  1  0 ⎥
⎣ 0  0  0  1 ⎦
```

**True T_EndEffector_to_Camera:**
```
⎡ 1  0  0  2 ⎤
⎢ 0  1  0  0 ⎥
⎢ 0  0  1  0 ⎥
⎣ 0  0  0  1 ⎦
```

---

#### Setup

Baseline = 30mm \
Working distance = 50mm

| Quantity                 | Camera 1        | Camera 2        |
|--------------------------|-----------------|-----------------|
| *Notation*               | *[x, y, z]*     | *[x, y, z]*     |
| End-effector position E  | [0, 0, 0]       | [**30**, 0, 0]  |
| Translation error t_err  | [2, 0, 0]       | [2,  0, 0]      |
| True camera center C     | [2, 0, 0]       | [**32**, 0, 0]  |
| Assumed camera center C' | [0, 0, 0]       | [**30**, 0, 0]  |
| True 3D point P          | [15, 0, **50**] | [15, 0, **50**] |


---

#### Step 1: What pixels do the cameras actually see?

The ray direction = endpoint - startpoint
                  = P - C

```
Camera 1:  ray direction = P - C₁ = [15, 0, 50] - [2,  0, 0]
Camera 2:  ray direction = P - C₂ = [15, 0, 50] - [32, 0, 0]

Ray 1 Direction = [13, 0, 50]
Ray 2 Direction = [-17, 0, 50]
```

These directions get projected to pixels. The pixel encodes the **direction**, not the camera center.

---

#### Step 2: Triangulation with wrong camera centers

We have the correct ray directions (from pixels), but wrong camera centers (C₁', C₂' instead of C₁, C₂).

```
Ray 1:  L₁(s₁) = C₁' + s₁ × r₁ = [0,  0, 0] + s₁ × normalize(Ray 1 Direction)
Ray 2:  L₂(s₂) = C₂' + s₂ × r₂ = [30, 0, 0] + s₂ × normalize(Ray 2 Direction)
```

Final ray equations:
```
L₁(s₁) = [0,  0, 0] + s₁ × [ 0.252, 0, 0.968]
L₂(s₂) = [30, 0, 0] + s₂ × [-0.322, 0, 0.947]
```

---

#### Step 3: Solve for intersection

The rays must intersect: L₁(s₁) = L₂(s₂)

Solving `L₁(s₁) = L₂(s₂)`:
```
[0,  0, 0] + s₁ × [ 0.252, 0, 0.968] = [30, 0, 0] + s₂ × [-0.322, 0, 0.947]

Solve for X-component: 0.252s₁ = 30 - 0.322s₂
Solve for Z-component: 0.968s₁ = 0.947s₂
```

Solving these equations:
```
s₂ = 52.77
s₁ = 51.6282
```

---

#### Step 4: Compute reconstructed point

```
P_reconstructed = L₁(s₁) = [0, 0, 0] + 51.6282 × [0.252, 0, 0.968]

P_reconstructed = [13.01, 0, 49.99]
```

---

#### Result

|                | Value          |
|----------------|----------------|
| True point     | [15, 0, 50]    |
| Reconstructed  | [13.01, 0, 49.99] |
| Error          | [-2, 0, 0] = -t_err |

**Conclusion:** Translation error shifts the entire reconstruction by -t_err. Depth and relative structure are preserved.

---

### Case 2: Rotation Error

|                       | Assumed       | True          |
|-----------------------|---------------|---------------|
| Rotation component    | I             | Ry(2°)        |

**Assumed T_EndEffector_to_Camera:**
```
⎡ 1  0  0  0 ⎤
⎢ 0  1  0  0 ⎥
⎢ 0  0  1  0 ⎥
⎣ 0  0  0  1 ⎦
```

**True T_EndEffector_to_Camera:**
```
⎡  0.9994  0  0.0349  0 ⎤
⎢    0     1    0     0 ⎥
⎢ -0.0349  0  0.9994  0 ⎥
⎣    0     0    0     1 ⎦
```

---

#### Setup

Baseline = 30mm \
Working distance = 50mm

| Quantity                 | Camera 1        | Camera 2        |
|--------------------------|-----------------|-----------------|
| *Notation*               | *[x, y, z]*     | *[x, y, z]*     |
| End-effector position E  | [0, 0, 0]       | [**30**, 0, 0]  |
| Camera center C          | [0, 0, 0]       | [**30**, 0, 0]  |
| True 3D point P          | [0, 0, **50**]  | [0, 0, **50**]  |


---

#### Step 1: What pixels do the cameras actually see?

![Inverse Projection Flowchart](/images/knowledge_base/concepts/computer_vision/camera_calibration/inverse_projection_flowchart.svg)

```
Ray 1 Direction (Camera Frame) = P - C₁ = [0, 0, 50] - [0,  0, 0]
Ray 2 Direction (Camera Frame) = P - C₂ = [0, 0, 50] - [30, 0, 0]

Ray 1 Direction (Camera Frame, normalized) = [ 0,     0, 1    ]
Ray 2 Direction (Camera Frame, normalized) = [-0.514, 0, 0.857]
```

```
Ray 1 Direction (World Frame, normalized) = Ry(-2°) @ [0, 0, 1]
Ray 2 Direction (World Frame, normalized) = Ry(-2°) @ [-0.514, 0, 0.857]

Ray 1 Direction (World Frame, normalized) = [-0.0349, 0, 0.999]
Ray 2 Direction (World Frame, normalized) = [-0.544,  0, 0.839]
```

---

#### Step 2: Triangulation with wrong ray directions

We have the correct camera centers, but wrong ray directions

```
Ray 1:  L₁(s₁) = C₁ + s₁ × r₁ = [0,  0, 0] + s₁ × [-0.0349, 0, 0.9994]
Ray 2:  L₂(s₂) = C₂ + s₂ × r₂ = [30, 0, 0] + s₂ × [-0.544,  0, 0.839]
```

---

#### Step 3: Solve for intersection

Solving `L₁(s₁) = L₂(s₂)`:
```
[0, 0, 0] + s₁ × [-0.0349, 0, 0.9994] = [30, 0, 0] + s₂ × [-0.544, 0, 0.839]

Solve for X-component: -0.0349s₁ = 30 - 0.544s₂
Solve for Z-component: 0.9994s₁ = 0.839s₂
```

Solving these equations:
```
s₂ = 52.32
s₁ = 43.92
```

---

#### Step 4: Compute reconstructed point

```
P_reconstructed = L₁(s₁) = [0, 0, 0] + 43.92 × [-0.0349, 0, 0.9994]

P_reconstructed = [-1.53, 0, 43.89]
```

---

#### Result

|                | Value                |
|----------------|----------------------|
| True point     | [0, 0, 50]           |
| Reconstructed  | [-1.53, 0, 43.89]    |
| Lateral error  | 1.53 mm              |
| Depth error    | 6.11 mm (12% error!) |
| Total error    | 6.30 mm              |

**Conclusion:** Rotation error causes **both lateral and depth errors**. Unlike translation, this **distorts the geometry** — relative distances between points are not preserved.

---

### Summary

| Error Type  | Magnitude | Lateral Error | Depth Error |
|-------------|-----------|---------------|-------------|
| Translation | 2 mm      | 2 mm          | 0 mm        |
| Rotation    | 2°        | 1.53 mm       | 6.11 mm     |

---

### Note on intrinsic matrix

The examples above skip K⁻¹ for simplicity. Here's how Case 1 would look with K⁻¹ incorporated:

Assume intrinsic matrix:
```
K = ⎡ 500  0  320 ⎤
    ⎢  0  500 240 ⎥
    ⎣  0   0   1  ⎦
```

**Step 1: What pixels do the cameras see?**

Ray directions (camera frame):
```
r₁ = P - C₁ = [13, 0, 50]
r₂ = P - C₂ = [-17, 0, 50]
```

Project to pixels: `pixel = K @ (r / r_z)`
```
pixel₁ = K @ [13/50, 0, 1]ᵀ = K @ [0.26, 0, 1]ᵀ = [450, 240]
pixel₂ = K @ [-17/50, 0, 1]ᵀ = K @ [-0.34, 0, 1]ᵀ = [150, 240]
```

**Step 2: Triangulation — recover rays from pixels**

Apply K⁻¹ to get normalized coordinates:
```
K⁻¹ = ⎡ 0.002   0   -0.64 ⎤
      ⎢   0   0.002  -0.48 ⎥
      ⎣   0     0      1   ⎦

normalized₁ = K⁻¹ @ [450, 240, 1]ᵀ = [0.26, 0, 1]
normalized₂ = K⁻¹ @ [150, 240, 1]ᵀ = [-0.34, 0, 1]
```

Ray directions (normalized):
```
r₁ = [0.26, 0, 1]   → [0.252, 0, 0.968]
r₂ = [-0.34, 0, 1]  → [-0.322, 0, 0.947]
```
Which is the same as what we started with!

**Key point:** K⁻¹ converts pixels → normalized image coordinates (ray directions with z=1). Since K is assumed correct, K and K⁻¹ cancel out — the ray directions we recover are the same as what we started with. This is why the examples skip K⁻¹.


