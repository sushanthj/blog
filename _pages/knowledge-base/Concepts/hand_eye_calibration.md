---
layout: knowledge
title: Hand-Eye Calibration
parent: Concepts
nav_order: 4
permalink: /knowledge-base/computer-vision/hand-eye-calibration/
toc: true
---

# Hand-Eye Calibration : Finding the Robot_End_Effector-to-Camera Transform

- Any robot with an end-effector (like the wrist of a robotic arm) usually has joints leading up to the end-effector.
- Encoders at these joints give joint angles. 
- These joint angles can be used to compute the pose of the end-effector in the robot's base frame in a process called 'forward kinematics'.
- Cameras are often mounted on the end-effector with an approximate known transform (T_end_effector_to_camera).
- However, due to tolerances/human error, this transform may not be exact. In this exercise we estimate the precise transform using hand-eye calibration.

## Methodology

We solve for the unknown transform `T_EndEffector_to_Camera` using synchronized pose and image data.

#### Nomenclature

| Symbol | Meaning |
|--------|---------|
| `T_robot_to_EndEffector[i]` | Transform from robot frame to end-effector (from forward kinematics, per image) |
| `T_camera_to_checkerboard[i]` | Transform from camera to checkerboard (computed via PnP, per image) |
| `T_EndEffector_to_Camera` | Transform from end-effector to camera (**unknown**, what we solve for) |
| `T_robot_to_checkerboard` | Transform from robot frame to checkerboard (**unknown constant**, checkerboard is fixed) |
| `@` | Matrix multiplication |

---

### High-Level Overview

1. **Setup**: Fix a checkerboard rigidly in the robot workspace
2. **Data collection**: Capture N images from diverse poses (non-coplanar, with end-effector rotations and tilt, and ensuring camera-sync), recording `T_robot_to_EndEffector[i]` with each image
3. **Checkerboard detection**: For each image, detect corners and use PnP to get `T_camera_to_checkerboard[i]`
4. **Hand-eye solve**: Use the AX=XB formulation to solve for `T_EndEffector_to_Camera`
5. **Refinement**: Optionally refine using reprojection error minimization
6. **Validation**: Verify via reprojection error and consistency checks

---

### Detailed Methodology

#### Problem Formulation

For each captured image `i`, we have:
- `T_robot_to_EndEffector[i]`: Robot Frame → End-Effector (from forward kinematics)
- `T_camera_to_checkerboard[i]`: Camera → Checkerboard (from PnP + intrinsics)

- `T_robot_to_checkerboard`: **Fixed** (but unknown)


So if we move just the end-effector (keeping Robot and Checkerboard fixed), at every timestamp `i` this will hold:
```
T_robot_to_checkerboard = Constant = T_robot_to_EndEffector[i] @ T_EndEffector_to_Camera @ T_camera_to_checkerboard[i]
```

#### Data Collection

1. **Fix a checkerboard** rigidly in the robot's workspace (must not move during calibration)
2. **Capture N images** (N ≥ 15 recommended)

#### Step 1: Compute `T_camera_to_checkerboard` via Perspective-n-Point (PnP) 

PnP finds the rotation `R` and translation `t` that minimize the difference between where the 3D points *should* project and where they *actually* appear. This gives us `T_camera_to_checkerboard`.

**Example Implementation (OpenCV):**
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

For any two poses `i` and `j`, `T_robot_to_checkerboard_i = T_robot_to_checkerboard_j = Constant`


`T_robot_to_checkerboard_i = T_robot_to_EndEffector[i] @ T_EndEffector_to_Camera @ T_camera_to_checkerboard[i]`
`T_robot_to_checkerboard_j = T_robot_to_EndEffector[j] @ T_EndEffector_to_Camera @ T_camera_to_checkerboard[j]`

Given they are equal:
```
T_robot_to_checkerboard_i = T_robot_to_checkerboard_j
```

Rearrange to form AX = XB:
```
inv(T_robot_to_EndEffector[j]) @ T_robot_to_EndEffector[i] @ T_EndEffector_to_Camera = T_EndEffector_to_Camera @ T_camera_to_checkerboard[j] @ inv(T_camera_to_checkerboard[i])
```

This is the **AX = XB** equation where:
- `A = inv(T_robot_to_EndEffector[j]) @ T_robot_to_EndEffector[i]` (relative end-effector motion)
- `B = T_camera_to_checkerboard[j] @ inv(T_camera_to_checkerboard[i])` (relative camera motion)
- `X = T_EndEffector_to_Camera` (unknown)

#### Step 3: Solve AX = XB

**Example Implementation (OpenCV):**
```python
R_EndEffector_to_Camera, t_EndEffector_to_Camera = cv2.calibrateHandEye(
    R_robot_to_EndEffector_list,   # list of rotation matrices
    t_robot_to_EndEffector_list,   # list of translation vectors
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

