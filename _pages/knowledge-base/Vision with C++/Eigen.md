---
layout: knowledge
title: Eigen Library
parent: Vision with C++
nav_order: 1
permalink: /knowledge-base/vision-cpp/eigen/
---

# Background

Eigen is the numpy equivalent in C++. Here we look at some basic linear algebra computations
using eigen.

[Credits:](https://aleksandarhaber.com/starting-with-eigen-c-matrix-library/)

## Installation

To install eigen3 we can use the apt repository.

- ```sudo apt update```
- ```sudo apt install libeigen3-dev```
- Verify your installation by doing ```dpkg -L libeigen3-dev```

Then to use in code you simply need to include the following header file and work within the
following namespace:

```cpp
#include <eigen3/Eigen/Dense>

using namespace Eigen;
```

# Declaring and Defining Matrices

## Basics

Here we'll define a 3x3 matrix in two equivalent ways:

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    // define 3x3 matrix -explicit declaration
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    // define 3x3 matrix -typedef declaration
    Matrix3f matrixA1;
    matrixA1.setZero();
    cout <<"\n"<<matrixA1<<endl;

    // Dynamic Allocation -explicit declaration
    Matrix <float, Dynamic, Dynamic> matrixB;

    // Dynamic Allocation -typedef declaration
    // 'X' denotes that the memory is to be dynamic
    MatrixXf matrixB1;

    // constructor method to declare matrix
    MatrixXd matrixC(10,10);

    // print any matrix in eigen is just piping to cout
    cout << endl << matrixC << endl;

    // resize any dynamic matrix
    MatrixXd matrixD1;
    matrixD1.resize(3, 3);
    matrixD1.setZero();
    cout << endl << matrixD1 << endl;

    return 0;
}
```

## Easier to Remember and Use

```cpp
int main()
{
    // directly init a matrix of zeros
    MatrixXf A;
    A = MatrixXf::Zero(3, 3);
    cout << "\n \n"<< A << endl;

    // directly init a matrix of ones
    MatrixXf B;
    B = MatrixXf::Ones(3, 3);
    cout << "\n \n"<< B << endl;

    // directly init a matrix filled with a constant value
    MatrixXf C;
    C = MatrixXf::Constant(3, 3, 1.2);
    cout << "\n \n"<< C << endl;

    // directly init identity (eye matrix)
    MatrixXd D;
    D = MatrixXd::Identity(3, 3);
    cout << "\n \n" << D << endl;

    MatrixXd E;
    E.setIdentity(3, 3);
    cout << "\n \n" << E << endl;
}
```

### Common Bug in above operations

```cpp
int main()
{
    MatrixXd V;
    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;
}
```

- If you try to run the above code it will compile. However, in execution it will segfault.
- The reason will be that we did not allocate memory for the matrix V.

We can fix this by doing the following:

```cpp
int main()
{
    MatrixXd V;
    // option 1
    V.resize(4,4);

    // option 2
    V = MatrixXd::Zero(4, 4);

    // best option
    MatrixXd V(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;
}
```

## Explicitly Defining Matrix Entries

We already saw this above, but once we have defined the right shape of the matrix,
we can then define it's entries as shown below:

**Note: Entries are interpreted in row-major order**

```cpp
MatrixXd V;
V.resize(4,4);

V << 101, 102, 103, 104,
    105, 106, 107, 108,
    109, 110, 111, 112,
    113, 114, 115, 116;
```

## Slicing Matrices

```cpp
int main()
{
    MatrixXd V = MatrixXd::Zero(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;

    MatrixXd Vblock = V.block(0, 0, 2, 2);
    cout << "\n \n" << Vblock << endl;
}
```

## Extracting Individual Rows and Columns + Find Shape

```cpp
int main()
{
    MatrixXd V = MatrixXd::Zero(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    MatrixXd row1 = V.row(0);
    MatrixXd col1 = V.col(0);

    cout << row1 << endl;
    cout << col1 << endl;
}
```

The above is useful in finding the shape of any given matrix (like numpy.shape)

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

int main() {
    MatrixXd matrix(3, 4);  // Example matrix with 3 rows and 4 columns

    int numRows = matrix.rows();
    int numCols = matrix.cols();

    std::cout << "Number of rows: " << numRows << std::endl;
    std::cout << "Number of columns: " << numCols << std::endl;

    return 0;
}
```

## Matrix Math

### Addition

```cpp
int main()
{
    MatrixXd A1(2, 2);
    MatrixXd B1(2, 2);

    A1 << 1, 2,
        3, 4;
    B1 << 3, 4,
        5, 6;
    MatrixXd C1 = A1 + B1;
    cout << " \n\n The sum of A1 and B1 is\n\n" << C1 << endl;
}
```

### Matrix Multiplication - Dot Prod and Scalar

Unlike numpy, here the * operator serves as default matrix multiplication

```cpp
int main()
{
    MatrixXd A1(2, 2);
    MatrixXd B1(2, 2);

    A1 << 1, 2,
        3, 4;
    B1 << 3, 4,
        5, 6;

    // Dot Product
    MatrixXd C1 = A1 * B1;

    // Multiplication by a scalar
    MatrixXd C2 = 2 * A1;
}
```

### Transpose

#### Don't do this

```cpp
int main()
{
    A1 = A1.transpose();
}
```

#### Can do this

```cpp
// the correct and safe way to do the matrix transpose is the following
A1.transposeInPlace();

// we can use a transpose operator in expressions
R1 = A1.transpose() + B1;
```