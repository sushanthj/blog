---
layout: knowledge
title: NumPy for Computer Vision
parent: Computer Vision
nav_order: 2
permalink: /knowledge-base/computer-vision/numpy/
---

# NumPy for Computer Vision

NumPy is a fundamental library for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices. This section covers essential NumPy operations for computer vision applications.

## Basic Operations

### Array Creation and Manipulation

```python
import numpy as np

# Create arrays
image = np.zeros((480, 640, 3), dtype=np.uint8)  # RGB image
kernel = np.ones((3, 3)) / 9  # 3x3 averaging kernel

# Array slicing
roi = image[100:200, 150:250]  # Region of interest
channel = image[:, :, 0]  # Red channel

# Reshaping
flattened = image.reshape(-1, 3)  # Flatten to Nx3 array
```

### Image Processing Operations

```python
# Convolution
def convolve2d(image, kernel):
    return np.sum(image * kernel, axis=(0, 1))

# Normalization
normalized = (image - image.min()) / (image.max() - image.min())

# Thresholding
binary = (image > 128).astype(np.uint8) * 255
```

## Advanced Techniques

### Feature Detection

```python
# Harris corner detection
def harris_corners(image, k=0.04):
    # Compute gradients
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)
    
    # Compute structure tensor
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Apply Gaussian filter
    window = np.ones((3, 3)) / 9
    Sxx = convolve2d(Ixx, window)
    Sxy = convolve2d(Ixy, window)
    Syy = convolve2d(Iyy, window)
    
    # Compute Harris response
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    harris_response = det - k * (trace ** 2)
    
    return harris_response
```

### Image Transformations

```python
# Affine transformation
def apply_affine(image, matrix):
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    coords = np.stack([x, y, np.ones_like(x)])
    transformed = matrix @ coords.reshape(3, -1)
    transformed = transformed.reshape(3, h, w)
    return transformed
```

## Performance Optimization

### Vectorization

```python
# Vectorized operations
def compute_gradients(image):
    # Use np.gradient for efficient computation
    return np.gradient(image)

# Broadcasting
def apply_kernel(image, kernel):
    # Use broadcasting for efficient convolution
    return np.sum(image * kernel[None, None, :, :], axis=(2, 3))
```

### Memory Management

```python
# Avoid unnecessary copies
def process_image(image):
    # Use views when possible
    view = image[::2, ::2]  # Creates a view, not a copy
    
    # Use in-place operations
    image *= 2  # Modifies in-place
    
    return image
```

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy for Image Processing](https://scipy-lectures.org/advanced/image_processing/)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/quickstart.html#advanced-indexing-and-index-tricks)