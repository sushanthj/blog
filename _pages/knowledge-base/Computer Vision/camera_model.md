---
layout: knowledge
title: Camera Models
parent: Computer Vision
nav_order: 1
permalink: /knowledge-base/computer-vision/camera-model/
---

# Camera Models and Calibration

This section covers the fundamental concepts of camera models and calibration in computer vision.

## Pinhole Camera Model

The pinhole camera model is the simplest and most widely used camera model in computer vision. It describes how 3D points in the world are projected onto a 2D image plane.

### Basic Principles

- Light rays pass through a single point (the pinhole)
- The image is formed on the image plane
- The distance between the pinhole and image plane is the focal length

### Mathematical Model

The projection of a 3D point $(X, Y, Z)$ to a 2D point $(u, v)$ can be described by:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 
\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} X/Z \\ Y/Z \\ 1 \end{bmatrix}
$$

Where:
- $f_x, f_y$ are the focal lengths
- $c_x, c_y$ are the principal point coordinates

## Lens Distortion

Real cameras have lenses that introduce distortion to the image. The two main types of distortion are:

### Radial Distortion

Radial distortion causes straight lines to appear curved. It can be modeled as:

$$
x_{distorted} = x(1 + k_1r^2 + k_2r^4 + k_3r^6)
$$

### Tangential Distortion

Tangential distortion occurs when the lens is not perfectly parallel to the image plane:

$$
x_{distorted} = x + [2p_1xy + p_2(r^2 + 2x^2)]
$$

## Camera Calibration

Camera calibration is the process of estimating the intrinsic and extrinsic parameters of a camera.

### Intrinsic Parameters

- Focal length
- Principal point
- Distortion coefficients

### Extrinsic Parameters

- Rotation matrix
- Translation vector

### Calibration Process

1. Capture multiple images of a calibration pattern
2. Detect pattern points in each image
3. Estimate camera parameters using optimization
4. Refine parameters to minimize reprojection error

## Implementation

```python
import cv2
import numpy as np

# Read calibration images
images = [cv2.imread(f'calibration_{i}.jpg') for i in range(10)]

# Find chessboard corners
pattern_size = (9, 6)
obj_points = []
img_points = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    
    if ret:
        obj_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        obj_points[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        img_points.append(corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)
```

## Additional Resources

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Camera Calibration Tools](https://github.com/opencv/opencv/tree/master/samples/python)