---
layout: knowledge
title: Optical Flow
parent: Computer Vision
nav_order: 3
permalink: /knowledge-base/computer-vision/optical-flow/
---

# Optical Flow

Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. This section covers the fundamental concepts and implementation of optical flow algorithms.

## Basic Concepts

### Brightness Constancy

The brightness constancy assumption states that the brightness of a pixel remains constant between consecutive frames:

```python
I(x, y, t) = I(x + dx, y + dy, t + dt)
```

### Lucas-Kanade Method

The Lucas-Kanade method assumes that the flow is constant in a small window around each pixel:

```python
import numpy as np

def lucas_kanade(frame1, frame2, window_size=15):
    # Compute gradients
    Ix = np.gradient(frame1, axis=1)
    Iy = np.gradient(frame1, axis=0)
    It = frame2 - frame1
    
    # Initialize flow fields
    u = np.zeros_like(frame1)
    v = np.zeros_like(frame1)
    
    # For each pixel
    for i in range(window_size, frame1.shape[0] - window_size):
        for j in range(window_size, frame1.shape[1] - window_size):
            # Extract window
            Ix_window = Ix[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            Iy_window = Iy[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            It_window = It[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            
            # Solve linear system
            A = np.vstack([Ix_window.flatten(), Iy_window.flatten()]).T
            b = -It_window.flatten()
            
            # Compute flow
            if np.linalg.det(A.T @ A) != 0:
                flow = np.linalg.solve(A.T @ A, A.T @ b)
                u[i, j] = flow[0]
                v[i, j] = flow[1]
    
    return u, v
```

## Advanced Techniques

### Pyramidal Lucas-Kanade

```python
def pyramidal_lucas_kanade(frame1, frame2, num_levels=3):
    # Build image pyramids
    pyramid1 = [frame1]
    pyramid2 = [frame2]
    
    for i in range(num_levels-1):
        pyramid1.append(cv2.pyrDown(pyramid1[-1]))
        pyramid2.append(cv2.pyrDown(pyramid2[-1]))
    
    # Compute flow at each level
    u = np.zeros_like(frame1)
    v = np.zeros_like(frame1)
    
    for level in range(num_levels-1, -1, -1):
        # Compute flow at current level
        u_level, v_level = lucas_kanade(pyramid1[level], pyramid2[level])
        
        # Upsample and add to previous flow
        if level < num_levels-1:
            u = cv2.pyrUp(u) + u_level
            v = cv2.pyrUp(v) + v_level
        else:
            u, v = u_level, v_level
    
    return u, v
```

## Applications

### Motion Detection

```python
def detect_motion(flow, threshold=0.5):
    # Compute flow magnitude
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    
    # Threshold to detect motion
    motion_mask = magnitude > threshold
    
    return motion_mask
```

### Object Tracking

```python
def track_object(frame1, frame2, bbox):
    # Extract region of interest
    x, y, w, h = bbox
    roi1 = frame1[y:y+h, x:x+w]
    roi2 = frame2[y:y+h, x:x+w]
    
    # Compute flow in ROI
    u, v = lucas_kanade(roi1, roi2)
    
    # Update bounding box
    dx = np.median(u)
    dy = np.median(v)
    
    return (x + dx, y + dy, w, h)
```

## Additional Resources

- [OpenCV Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Lucas-Kanade Method Paper](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf)
- [Optical Flow Estimation](https://www.cs.cornell.edu/~dph/papers/flow-cvpr07.pdf)