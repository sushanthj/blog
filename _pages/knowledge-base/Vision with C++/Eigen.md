---
layout: knowledge
title: Eigen Library
parent: Vision with C++
nav_order: 1
permalink: /knowledge-base/vision-cpp/eigen/
---

# Eigen Library

Eigen is a C++ template library for linear algebra that provides efficient implementations of matrices, vectors, and numerical algorithms. This section covers the essential features of Eigen for computer vision applications.

## Basic Usage

### Matrix and Vector Types

```cpp
#include <Eigen/Dense>

// Fixed-size matrices and vectors
Eigen::Matrix3d rotation_matrix;  // 3x3 double matrix
Eigen::Vector4d homogeneous_point;  // 4x1 double vector

// Dynamic-size matrices and vectors
Eigen::MatrixXd dynamic_matrix(rows, cols);  // rows x cols double matrix
Eigen::VectorXf dynamic_vector(size);  // size x 1 float vector
```

### Common Operations

```cpp
// Matrix operations
Eigen::Matrix3d A, B, C;
C = A * B;  // Matrix multiplication
C = A + B;  // Matrix addition
C = A.transpose();  // Matrix transpose

// Vector operations
Eigen::Vector3d v1, v2;
double dot_product = v1.dot(v2);  // Dot product
Eigen::Vector3d cross_product = v1.cross(v2);  // Cross product
```

## Computer Vision Applications

### Camera Calibration

```cpp
// Camera matrix
Eigen::Matrix3d K;  // Intrinsic parameters
K << fx, 0, cx,
     0, fy, cy,
     0, 0, 1;

// Rotation matrix
Eigen::Matrix3d R;  // Extrinsic rotation
Eigen::Vector3d t;  // Extrinsic translation

// Projection
Eigen::Vector3d point_3d;
Eigen::Vector2d point_2d = (K * (R * point_3d + t)).hnormalized();
```

### 3D Transformations

```cpp
// Rotation matrix from angle-axis
Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
double angle = M_PI / 4;
Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();

// Homogeneous transformation
Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
T.block<3,3>(0,0) = R;
T.block<3,1>(0,3) = t;
```

## Performance Optimization

### Memory Management

```cpp
// Avoid unnecessary copies
Eigen::MatrixXd A = Eigen::MatrixXd::Random(1000, 1000);
Eigen::MatrixXd B = A;  // Creates a copy
Eigen::MatrixXd& C = A;  // Creates a reference

// Use noalias() for in-place operations
A.noalias() = B * C;  // Avoids temporary allocation
```

### Compiler Optimization

```cpp
// Enable vectorization
#pragma omp simd
for(int i = 0; i < size; ++i) {
    result[i] = a[i] * b[i];
}

// Use aligned memory
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
```

## Additional Resources

- [Eigen Documentation](https://eigen.tuxfamily.org/dox/)
- [Eigen Quick Reference Guide](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
- [Eigen Performance Tips](https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html)