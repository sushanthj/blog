---
layout: knowledge
title: Eigen Applications in Computer Vision
parent: Vision with C++
nav_order: 2
permalink: /knowledge-base/vision-cpp/eigen-applied/
---

# Eigen Applications in Computer Vision

This section demonstrates practical applications of Eigen in computer vision tasks, focusing on common operations and algorithms.

## Image Processing

### Basic Image Operations

```cpp
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// Convert OpenCV Mat to Eigen Matrix
Eigen::MatrixXd cvMatToEigen(const cv::Mat& mat) {
    Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
    for(int i = 0; i < mat.rows; ++i) {
        for(int j = 0; j < mat.cols; ++j) {
            eigen_mat(i,j) = mat.at<double>(i,j);
        }
    }
    return eigen_mat;
}

// Convert Eigen Matrix to OpenCV Mat
cv::Mat eigenToCvMat(const Eigen::MatrixXd& eigen_mat) {
    cv::Mat mat(eigen_mat.rows(), eigen_mat.cols(), CV_64F);
    for(int i = 0; i < eigen_mat.rows(); ++i) {
        for(int j = 0; j < eigen_mat.cols(); ++j) {
            mat.at<double>(i,j) = eigen_mat(i,j);
        }
    }
    return mat;
}
```

## Feature Detection and Matching

### Homography Estimation

```cpp
// Estimate homography using RANSAC
Eigen::Matrix3d estimateHomography(const std::vector<Eigen::Vector2d>& src_points,
                                 const std::vector<Eigen::Vector2d>& dst_points) {
    // Normalize points
    Eigen::Matrix3d T1 = normalizePoints(src_points);
    Eigen::Matrix3d T2 = normalizePoints(dst_points);
    
    // Compute homography using DLT
    Eigen::MatrixXd A(2 * src_points.size(), 9);
    for(size_t i = 0; i < src_points.size(); ++i) {
        Eigen::Vector2d p1 = (T1 * src_points[i].homogeneous()).hnormalized();
        Eigen::Vector2d p2 = (T2 * dst_points[i].homogeneous()).hnormalized();
        
        A.row(2*i) << p1.x(), p1.y(), 1, 0, 0, 0, -p2.x()*p1.x(), -p2.x()*p1.y(), -p2.x();
        A.row(2*i+1) << 0, 0, 0, p1.x(), p1.y(), 1, -p2.y()*p1.x(), -p2.y()*p1.y(), -p2.y();
    }
    
    // Solve using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d H;
    H << h(0), h(1), h(2),
         h(3), h(4), h(5),
         h(6), h(7), h(8);
    
    // Denormalize
    return T2.inverse() * H * T1;
}
```

## 3D Reconstruction

### Triangulation

```cpp
// Triangulate 3D point from two views
Eigen::Vector3d triangulatePoint(const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
                               const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
                               const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
    // Build the DLT matrix
    Eigen::Matrix4d A;
    A.row(0) = p1.x() * R1.row(2) - R1.row(0);
    A.row(1) = p1.y() * R1.row(2) - R1.row(1);
    A.row(2) = p2.x() * R2.row(2) - R2.row(0);
    A.row(3) = p2.y() * R2.row(2) - R2.row(1);
    
    // Solve using SVD
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    
    return X.head<3>() / X(3);
}
```

## Bundle Adjustment

### Cost Function

```cpp
// Bundle adjustment cost function using Eigen
struct BundleAdjustmentCost {
    BundleAdjustmentCost(const Eigen::Vector2d& observation)
        : observation_(observation) {}
    
    template <typename T>
    bool operator()(const T* const camera,
                   const T* const point,
                   T* residuals) const {
        // Project 3D point to image plane
        Eigen::Map<const Eigen::Matrix<T,3,1>> R(camera);
        Eigen::Map<const Eigen::Matrix<T,3,1>> t(camera + 3);
        Eigen::Map<const Eigen::Matrix<T,3,1>> X(point);
        
        // Compute projection
        Eigen::Matrix<T,3,1> p = R * X + t;
        T x = p(0) / p(2);
        T y = p(1) / p(2);
        
        // Compute residuals
        residuals[0] = x - T(observation_.x());
        residuals[1] = y - T(observation_.y());
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation) {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentCost, 2, 6, 3>(
            new BundleAdjustmentCost(observation));
    }
    
    const Eigen::Vector2d observation_;
};
```

## Additional Resources

- [Eigen with OpenCV](https://docs.opencv.org/master/d1/d1a/namespacecv.html)
- [Bundle Adjustment Tutorial](https://ceres-solver.org/tutorial.html)
- [3D Reconstruction Resources](https://www.robots.ox.ac.uk/~vgg/hzbook/)