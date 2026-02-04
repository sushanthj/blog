## Extrinsics Calibration : Finding the Shape Sensor-to-Camera Transform

The shape sensor terminates at the distal tip of the endoscope. The camera is placed at a fixed offset from the distal tip of the endoscope.

However, due to manufacturing tolerances, the camera may be placed with a tilt in viewing direction or a small offset in the x or y direction. Here we estimate the transform between the shape sensor and the camera using a calibration pattern with known dimensions.

### Methodology

This is a variant of the classic **hand-eye calibration** problem. We solve for the unknown transform `T_shapetip_to_camera` using synchronized pose and image data.

#### Nomenclature

| Symbol | Meaning |
|--------|---------|
| `T_robot_to_shapetip[i]` | Transform from robot frame to shape sensor tip (from shape sensing, per image) |
| `T_camera_to_checkerboard[i]` | Transform from camera to checkerboard (computed via PnP, per image) |
| `T_shapetip_to_camera` | Transform from shape sensor tip to camera (**unknown**, what we solve for) |
| `T_robot_to_checkerboard` | Transform from robot frame to checkerboard (**unknown constant**, checkerboard is fixed) |
| `@` | Matrix multiplication |

---

### High-Level Overview

1. **Setup**: Fix a checkerboard rigidly in the robot workspace
2. **Data collection**: Capture N images from diverse poses (non-coplanar, with endoscope rotations and tilt, and ensuring camera-shape sync), recording `T_robot_to_shapetip[i]` with each image
3. **Checkerboard detection**: For each image, detect corners and use PnP to get `T_camera_to_checkerboard[i]`
4. **Hand-eye solve**: Use the AX=XB formulation to solve for `T_shapetip_to_camera`
5. **Refinement**: Optionally refine using reprojection error minimization
6. **Validation**: Verify via reprojection error and consistency checks

---

### Detailed Methodology

#### Problem Formulation

For each captured image `i`, we have:
- `T_robot_to_shapetip[i]`: Robot Frame → Shape Sensor Tip (from shape sensing)
- `T_camera_to_checkerboard[i]`: Camera → Checkerboard (from PnP + intrinsics)

- `T_robot_to_checkerboard`: Constant (but unknown)


So if we move just the endoscope (keeping Robot and Checkerboard fixed), at every timestamp `i` this will hold:
```
T_robot_to_checkerboard = T_robot_to_shapetip[i] @ T_shapetip_to_camera @ T_camera_to_checkerboard[i]
```

#### Data Collection

1. **Fix a checkerboard** rigidly in the robot's workspace (must not move during calibration)
2. **Capture N images** (N ≥ 15 recommended) from diverse poses:
   - Vary translation in x, y, z
   - Vary rotation (tilt the endoscope at different angles)
   - Ensure the checkerboard is fully visible and in focus
   - Ensure camera-shape sync
3. **Record synchronized data** for each image:
   - The image itself
   - The corresponding `T_robot_to_shapetip[i]` at capture time

#### Step 1: Compute Camera-to-Checkerboard Transforms via PnP

**What is PnP?**

Perspective-n-Point (PnP) solves for the pose of a calibrated camera given:
- A set of 3D points in a known coordinate frame (checkerboard corners in checkerboard frame)
- Their corresponding 2D projections in the image (detected corner pixel locations)
- The camera intrinsic matrix K and distortion coefficients

PnP finds the rotation `R` and translation `t` that minimize the difference between where the 3D points *should* project and where they *actually* appear. This gives us `T_camera_to_checkerboard`.

**Implementation:**
```python
ret, corners = cv2.findChessboardCorners(image, pattern_size)
corners_refined = cv2.cornerSubPix(gray, corners, ...)
success, rvec, tvec = cv2.solvePnP(object_points, corners_refined, K, dist_coeffs)
R, _ = cv2.Rodrigues(rvec)
T_camera_to_checkerboard = np.eye(4)
T_camera_to_checkerboard[:3, :3] = R
T_camera_to_checkerboard[:3, 3] = tvec.flatten()
```

#### Step 2: Form the AX = XB Hand-Eye Problem

For any two poses `i` and `j`, T_robot_to_checkerboard_i = T_robot_to_checkerboard_j

```
T_robot_to_checkerboard_i = T_robot_to_shapetip[i] @ T_shapetip_to_camera @ T_camera_to_checkerboard[i]
T_robot_to_checkerboard_j = T_robot_to_shapetip[j] @ T_shapetip_to_camera @ T_camera_to_checkerboard[j]
```

Final equation:
```
T_robot_to_checkerboard_i = T_robot_to_checkerboard_j
```

Rearrange to form AX = XB:
```
inv(T_robot_to_shapetip[j]) @ T_robot_to_shapetip[i] @ T_shapetip_to_camera = T_shapetip_to_camera @ T_camera_to_checkerboard[j] @ inv(T_camera_to_checkerboard[i])
```

This is the **AX = XB** equation where:
- `A = inv(T_robot_to_shapetip[j]) @ T_robot_to_shapetip[i]` (relative shape sensor motion)
- `B = T_camera_to_checkerboard[j] @ inv(T_camera_to_checkerboard[i])` (relative camera motion)
- `X = T_shapetip_to_camera` (unknown)

#### Step 3: Solve AX = XB

Use OpenCV's solver:
```python
R_shapetip_to_camera, t_shapetip_to_camera = cv2.calibrateHandEye(
    R_robot_to_shapetip_list,   # list of rotation matrices
    t_robot_to_shapetip_list,   # list of translation vectors
    R_camera_to_checkerboard_list,
    t_camera_to_checkerboard_list,
    method=cv2.CALIB_HAND_EYE_TSAI  # or PARK, HORAUD, ANDREFF, DANIILIDIS
)
```

#### Step 4: Refine with Reprojection Error Minimization (Optional)

**What is reprojection error?**

Reprojection error measures how well the estimated transforms predict the observed image. For each checkerboard corner:
1. Take its known 3D position in checkerboard frame
2. Transform it through the full kinematic chain to camera frame
3. Project it to 2D using the camera intrinsics
4. Compare to the *actually detected* 2D corner location

```
reprojection_error = || projected_2D - detected_2D ||
```

**Optimization objective:**
```
minimize sum_i sum_j || project(T_robot_to_shapetip[i] @ T_shapetip_to_camera @ P_checkerboard[j]) - detected_corners[i,j] ||^2
```

Where `P_checkerboard[j]` is the j-th checkerboard corner in checkerboard coordinates (known), and optimization is over `T_shapetip_to_camera` and `T_robot_to_checkerboard`.

Use the hand-eye solution as initialization. This nonlinear refinement directly optimizes for image-space accuracy.

---

### Validation

1. **Reprojection error**: Project checkerboard corners using estimated `T_shapetip_to_camera` and compare to detected corners. Target: **< 1 pixel RMS**
2. **Consistency check**: Compute `T_robot_to_checkerboard` from each image independently; they should all agree within noise
3. **Cross-validation**: Hold out some images, calibrate on the rest, verify reprojection on held-out data

---

### Tips for Good Calibration

- **Pose diversity is critical**: Avoid coplanar or near-parallel motions between poses
- **Use your rough estimate**: Initialize nonlinear optimization with your existing `T_shapetip_to_camera` estimate
- **Check for outliers**: Remove images with poor checkerboard detection or timing sync issues
- **Intrinsics must be accurate**: Errors in camera intrinsics propagate directly to extrinsics
